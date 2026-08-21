import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import json
import threading
import collections
import ctypes
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def lerp(current, target, alpha=0.3):
    return current + alpha * (target - current)

def overlay_transparent(background, overlay, x, y):
    x, y = int(x), int(y)
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    y1, y2 = max(0, y), min(bg_h, y + h)
    x1, x2 = max(0, x), min(bg_w, x + w)
    y1_o, y2_o = max(0, -y), h - max(0, (y + h) - bg_h)
    x1_o, x2_o = max(0, -x), w - max(0, (x + w) - bg_w)

    alpha_s = overlay[y1_o:y2_o, x1_o:x2_o, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            alpha_s * overlay[y1_o:y2_o, x1_o:x2_o, c] +
            alpha_l * background[y1:y2, x1:x2, c]
        )
    return background

def draw_rounded_rect(img, pt1, pt2, color, thickness=1, radius=15):
    """Draws a sleek rounded rectangle card for modern UI elements."""
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 - x1 < 2 * radius: x2 = x1 + 2 * radius
    if y2 - y1 < 2 * radius: y2 = y1 + 2 * radius
    
    if thickness == cv2.FILLED:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, cv2.FILLED)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, cv2.FILLED)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, cv2.FILLED)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, cv2.FILLED)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),            # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm Base
]

# ==========================================
# ENGINE CLASSES
# ==========================================
class WebcamStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            try:
                self.ret, self.frame = self.cap.read()
            except Exception as e:
                print(f"Camera read error: {e}")

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

class Particle:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius = 10
        self.life = 15

    def update(self):
        self.radius += 8
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            cv2.circle(surface, (int(self.x), int(self.y)), self.radius, (0, 255, 0), 3)

class Component:
    def __init__(self, config):
        self.id = config["id"]
        self.w, self.h = config["w"], config["h"]
        self.start_x, self.start_y = config["start_x"], config["start_y"]
        self.target_x, self.target_y = config["target_x"], config["target_y"]
        
        self.x, self.y = float(self.start_x), float(self.start_y)
        self.target_render_x, self.target_render_y = float(self.start_x), float(self.start_y)
        self.snapped = False

        self.img = cv2.imread(config["img_path"], cv2.IMREAD_UNCHANGED)
        if self.img is not None:
            self.img = cv2.resize(self.img, (self.w, self.h))

    def update(self):
        self.x = lerp(self.x, self.target_render_x, 0.4)
        self.y = lerp(self.y, self.target_render_y, 0.4)

    def draw_socket(self, surface, is_grabbed, current_time):
        if not self.snapped:
            if is_grabbed:
                pulse = (math.sin(current_time * 8) + 1) / 2
                glow_intensity = int(100 + (155 * pulse))
                glow_color = (0, 255, glow_intensity)
                
                cv2.rectangle(surface, (self.target_x, self.target_y), 
                            (self.target_x + self.w, self.target_y + self.h), 
                            glow_color, 4)
                cv2.rectangle(surface, (self.target_x - 6, self.target_y - 6), 
                            (self.target_x + self.w + 6, self.target_y + self.h + 6), 
                            glow_color, 1)
            else:
                cv2.rectangle(surface, (self.target_x, self.target_y), 
                            (self.target_x + self.w, self.target_y + self.h), 
                            (0, 200, 255), 2)

    def draw_component(self, surface, is_grabbed, debug=False):
        if is_grabbed:
            shadow_offset = 15
            if self.img is not None and self.img.shape[2] == 4:
                shadow_img = np.zeros((self.h, self.w, 4), dtype=np.uint8)
                shadow_img[:, :, :3] = (10, 10, 10)  
                shadow_img[:, :, 3] = (self.img[:, :, 3] * 0.5).astype(np.uint8)  
                overlay_transparent(surface, shadow_img, self.x + shadow_offset, self.y + shadow_offset)
            else:
                cv2.rectangle(surface, 
                            (int(self.x) + shadow_offset, int(self.y) + shadow_offset), 
                            (int(self.x + self.w) + shadow_offset, int(self.y + self.h) + shadow_offset), 
                            (10, 10, 10), -1)
                            
            cv2.rectangle(surface, (int(self.x)-2, int(self.y)-2), 
                        (int(self.x + self.w)+2, int(self.y + self.h)+2), 
                        (0, 255, 0), 3)

        if self.img is not None:
            overlay_transparent(surface, self.img, self.x, self.y)
        
        if debug:
            cv2.rectangle(surface, (int(self.x), int(self.y)), 
                        (int(self.x + self.w), int(self.y + self.h)), (255, 0, 255), 1)

# ==========================================
# GAME MANAGER
# ==========================================
class CircuitBuilderGame:
    def setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        
        dummy_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.imshow(self.window_name, dummy_frame)
        cv2.waitKey(100)  

        hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
        if hwnd:
            try:
                ctypes.windll.user32.ShowWindow(hwnd, 9) 
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                ctypes.windll.user32.SetFocus(hwnd)
            except Exception as e:
                print(f"Focus error: {e}")
    
    def __init__(self):
        self.w, self.h = 1280, 720
        self.window_name = "Plexus Circuit Builder"

        self.game_state = "PLAYING"
        
        self.components = []
        self.particles = []
        self.grabbed_idx = None
        self.cursor_x, self.cursor_y = self.w / 2, self.h / 2
        
        self.is_pinching = False
        self.PINCH_GRAB_RATIO = 0.20
        self.PINCH_DROP_RATIO = 0.45
        self.PINCH_OPEN_RATIO = 1.1  
        self.pinch_ratio = self.PINCH_OPEN_RATIO
        self.hand_visible = False

        self.show_camera_panel = True
        
        self.show_debug = False
        self.fps_history = collections.deque(maxlen=60)
        self.inf_history = collections.deque(maxlen=60)
        self.confidence_score = 0.0

        self.load_config()
        self.setup_window()
        self.stream = WebcamStream(src=0, width=self.w, height=self.h)
        
        self.latest_result = None
        self.inference_lock = threading.Lock()
        self.inference_running = True
        
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task', delegate=python.BaseOptions.Delegate.CPU)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, running_mode=vision.RunningMode.IMAGE)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()

    def inference_worker(self):
        while self.inference_running:
            try:
                success, frame = self.stream.read()
                if not success or frame is None:
                    time.sleep(0.01)
                    continue
                    
                frame = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                
                start_t = time.perf_counter()
                result = self.detector.detect(mp_image)
                inf_time = (time.perf_counter() - start_t) * 1000 
                
                with self.inference_lock:
                    self.latest_result = result
                    self.inf_history.append(inf_time)
                    if result.hand_landmarks:
                        self.confidence_score = getattr(result.handedness[0][0], 'score', 1.0)
                    else:
                        self.confidence_score = 0.0
            except Exception as e:
                pass
                
            time.sleep(0.001)

    def load_config(self):
        with open("components.json", "r") as f:
            config = json.load(f)
        pcb_conf = config["pcb"]
        raw_board = cv2.imread(pcb_conf["image"])

        max_w, max_h = 750, 560
        orig_h, orig_w = raw_board.shape[:2]
        scale = min(max_w / orig_w, max_h / orig_h)
        self.new_w, self.new_h = int(orig_w * scale), int(orig_h * scale)
        self.board_img = cv2.resize(raw_board, (self.new_w, self.new_h))

        self.board_x = 50
        self.board_y = (self.h - self.new_h) // 2 + 40

        self.panel_x = self.board_x + self.new_w + 40
        self.panel_w = (self.w - 40) - self.panel_x
        self.panel_y = self.board_y
        self.panel_h = self.new_h

        for comp_data in config["components"]:
            self.components.append(Component(comp_data))

    def draw_debug_hud(self, surface):
        if not self.show_debug: return
        
        hud_w, hud_h = 300, 190
        cv2.rectangle(surface, (10, 10), (10 + hud_w, 10 + hud_h), (0, 0, 0, 150), -1)
        cv2.putText(surface, "[DEBUG TELEMETRY]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        current_fps = self.fps_history[-1] if self.fps_history else 0
        current_inf = self.inf_history[-1] if self.inf_history else 0
        
        cv2.putText(surface, f"Render FPS: {int(current_fps)}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(surface, f"Infer Latency: {int(current_inf)}ms", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(surface, f"Hand Conf: {self.confidence_score:.2f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(surface, f"Pinch State: {self.is_pinching}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(surface, f"Pinch Ratio: {self.pinch_ratio:.2f} (grab<{self.PINCH_GRAB_RATIO} drop>{self.PINCH_DROP_RATIO})",
                    (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if len(self.fps_history) > 1:
            pts = []
            for i, fps_val in enumerate(self.fps_history):
                x = 20 + int((i / 60) * 280)
                y = 185 - int((min(fps_val, 120) / 120) * 30)
                pts.append((x, y))
            cv2.polylines(surface, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), 1)

    def draw_header(self, display):
        draw_rounded_rect(display, (50, 15), (self.w - 50, 70), (30, 38, 52), cv2.FILLED, 10)
        draw_rounded_rect(display, (50, 15), (self.w - 50, 70), (60, 80, 110), 1, 10)
        
        cv2.putText(display, "PLEXUS CIRCUIT BUILDER", (75, 52), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
        
        instruction_text = "Pinch to grab & snap" if not self.is_pinching else "Dragging component..."
        cv2.putText(display, instruction_text, (430, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 210, 225), 1)
        
        coord_text = f"X: {int(self.cursor_x)}  Y: {int(self.cursor_y)}"
        draw_rounded_rect(display, (self.w - 240, 28), (self.w - 70, 58), (15, 20, 30), cv2.FILLED, 6)
        draw_rounded_rect(display, (self.w - 240, 28), (self.w - 70, 58), (70, 85, 105), 1, 6)
        cv2.putText(display, coord_text, (self.w - 225, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_operator_panel(self, display, results):
        if not self.show_camera_panel: return
        
        panel_x, panel_y, panel_w, panel_h = self.panel_x, self.panel_y, self.panel_w, self.panel_h

        draw_rounded_rect(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 38, 52), cv2.FILLED, 15)
        draw_rounded_rect(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (60, 80, 110), 2, 15)

        text_x = panel_x + 30
        text_y = panel_y + 50
        cv2.putText(display, "COMPONENT TRAY", (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        card_w = panel_w - 60
        pip_h = min(int(card_w * (9 / 16)), 260)
        pip_w = int(pip_h * (16 / 9))
        pip_x = text_x + (card_w - pip_w) // 2
        panel_bottom = panel_y + panel_h - 20
        pip_y = panel_bottom - pip_h

        gauge_y2 = pip_y - 40
        gauge_y1 = gauge_y2 - 30
        tray_y1 = text_y + 30
        tray_y2 = gauge_y1 - 40

        if tray_y2 > tray_y1 + 20:
            draw_rounded_rect(display, (text_x, tray_y1), (text_x + card_w, tray_y2), (22, 28, 38), cv2.FILLED, 10)
            draw_rounded_rect(display, (text_x, tray_y1), (text_x + card_w, tray_y2), (70, 85, 105), 1, 10)
            tray_label = "Components start here"
            label_size = cv2.getTextSize(tray_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.putText(display, tray_label, (text_x + (card_w - label_size[0]) // 2, (tray_y1 + tray_y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 155, 175), 1)

        # --- LIVE PINCH GAUGE ---
        draw_rounded_rect(display, (text_x, gauge_y1), (text_x + card_w, gauge_y1 + 30), (15, 20, 30), cv2.FILLED, 8)
        fill_ratio = 0.0
        if self.hand_visible:
            span = max(self.PINCH_OPEN_RATIO - self.PINCH_GRAB_RATIO, 1e-4)
            fill_ratio = (self.PINCH_OPEN_RATIO - self.pinch_ratio) / span
            fill_ratio = max(0.0, min(1.0, fill_ratio))
        fill_w = int(card_w * fill_ratio)
        gauge_color = (0, 255, 0) if self.is_pinching else (0, 165, 255)
        
        if fill_w > 16:
            draw_rounded_rect(display, (text_x, gauge_y1), (text_x + fill_w, gauge_y1 + 30), gauge_color, cv2.FILLED, 8)
        draw_rounded_rect(display, (text_x, gauge_y1), (text_x + card_w, gauge_y1 + 30), (90, 110, 140), 1, 8)

        gauge_label = "LOCKED - GRABBING" if self.is_pinching else ("PINCH LOCK" if self.hand_visible else "NO HAND DETECTED")
        cv2.putText(display, gauge_label, (text_x, gauge_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        self.draw_pip(display, results, pip_x, pip_y, pip_w, pip_h)

    def draw_pip(self, display, results, pip_x, pip_y, pip_w, pip_h):
        success_pip, pip_frame = self.stream.read()
        if success_pip and pip_frame is not None:
            pip_resized = cv2.resize(cv2.flip(pip_frame, 1), (pip_w, pip_h))

            if results and results.hand_landmarks:
                lms = results.hand_landmarks[0]
                for start_idx, end_idx in HAND_CONNECTIONS:
                    x1, y1 = int(lms[start_idx].x * pip_w), int(lms[start_idx].y * pip_h)
                    x2, y2 = int(lms[end_idx].x * pip_w), int(lms[end_idx].y * pip_h)
                    cv2.line(pip_resized, (x1, y1), (x2, y2), (255, 255, 255), 2)
                for lm_pt in lms:
                    cx, cy = int(lm_pt.x * pip_w), int(lm_pt.y * pip_h)
                    cv2.circle(pip_resized, (cx, cy), 4, (0, 215, 255), -1)
                
                # Highlight the pinching fingers in the PiP
                tx, ty = int(lms[4].x * pip_w), int(lms[4].y * pip_h)
                ix, iy = int(lms[8].x * pip_w), int(lms[8].y * pip_h)
                pinch_color = (0, 255, 0) if self.is_pinching else (0, 220, 255)
                cv2.line(pip_resized, (tx, ty), (ix, iy), pinch_color, 2)
                cv2.circle(pip_resized, (tx, ty), 6, pinch_color, -1)
                cv2.circle(pip_resized, (ix, iy), 6, pinch_color, -1)

            draw_rounded_rect(display, (pip_x - 3, pip_y - 3), (pip_x + pip_w + 3, pip_y + pip_h + 3), (100, 120, 150), 2, 8)
            display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = pip_resized

            if not self.hand_visible:
                warn = display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w].copy()
                cv2.rectangle(warn, (0, 0), (pip_w, pip_h), (0, 0, 0), cv2.FILLED)
                cv2.addWeighted(warn, 0.55, display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w], 0.45, 0,
                                display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w])
                cv2.putText(display, "NO HAND DETECTED", (pip_x + 20, pip_y + pip_h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 100, 255), 2)

        cv2.putText(display, "LIVE CAMERA FEED", (pip_x, pip_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def run(self):
        prev_time = time.perf_counter()
        try:
            while True:
                with self.inference_lock:
                    results = self.latest_result
                
                # 1. Create blank canvas exactly ONCE
                display = np.full((self.h, self.w, 3), (25, 25, 25), dtype=np.uint8)
                
                # 2. Draw the PCB Board
                display[self.board_y:self.board_y+self.new_h, self.board_x:self.board_x+self.new_w] = self.board_img

                target_cursor_x, target_cursor_y = self.cursor_x, self.cursor_y
                cursor_active = False

                if results and results.hand_landmarks:
                    cursor_active = True
                    lm = results.hand_landmarks[0]
                    
                    margin_x, margin_y = 0.20, 0.20 
                    def map_coord(val, margin, max_val):
                        mapped = (val - margin) / (1.0 - 2.0 * margin)
                        return max(0.0, min(float(max_val), mapped * max_val))
                    
                    # STABLE CURSOR FIX: Track the Knuckle (Landmark 9) for rock-solid stability
                    target_cursor_x = map_coord(lm[9].x, margin_x, self.w)
                    target_cursor_y = map_coord(lm[9].y, margin_y, self.h)

                    # Dynamic pinch logic using Thumb and Index distances
                    wrist, mid_mcp = lm[0], lm[9]
                    hand_scale = max(math.hypot(wrist.x - mid_mcp.x, wrist.y - mid_mcp.y), 1e-4)
                    raw_pinch_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
                    self.pinch_ratio = raw_pinch_dist / hand_scale

                    if not self.is_pinching and self.pinch_ratio < self.PINCH_GRAB_RATIO:
                        self.is_pinching = True
                    elif self.is_pinching and self.pinch_ratio > self.PINCH_DROP_RATIO:
                        self.is_pinching = False

                self.hand_visible = cursor_active
                self.cursor_x = lerp(self.cursor_x, target_cursor_x, 0.5)
                self.cursor_y = lerp(self.cursor_y, target_cursor_y, 0.5)

                self.handle_logic()

                current_time = time.perf_counter()

                # --- DRAW CLEAN UI OVERLAYS ---
                self.draw_header(display)
                self.draw_operator_panel(display, results)

                # --- DRAW GAME ELEMENTS ---
                for i, comp in enumerate(self.components):
                    comp.draw_socket(display, is_grabbed=(self.grabbed_idx == i), current_time=current_time)

                for i, comp in enumerate(self.components):
                    comp.update()
                    comp.draw_component(display, is_grabbed=(self.grabbed_idx == i), debug=self.show_debug)

                for p in self.particles[:]:
                    p.update()
                    p.draw(display)
                    if p.life <= 0: self.particles.remove(p)

                if self.game_state == "WIN":
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (self.w, self.h), (10, 20, 15), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

                    draw_rounded_rect(display, (self.w//2 - 380, self.h//2 - 160), (self.w//2 + 380, self.h//2 + 140), (25, 45, 35), cv2.FILLED, 16)
                    draw_rounded_rect(display, (self.w//2 - 380, self.h//2 - 160), (self.w//2 + 380, self.h//2 + 140), (0, 255, 128), 2, 16)

                    cv2.putText(display, "CIRCUIT COMPLETE!", (self.w//2 - 250, self.h//2 - 80), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 128), 2)
                    cv2.putText(display, "All components successfully installed.", (self.w//2 - 260, self.h//2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1)
                    cv2.putText(display, "Press 'R' to Build Again  |  Press 'ESC' to Exit", (self.w//2 - 260, self.h//2 + 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 170, 170), 1)

                # --- DRAW CURSOR ---
                if cursor_active:
                    if self.is_pinching:
                        cv2.circle(display, (int(self.cursor_x), int(self.cursor_y)), 12, (0, 255, 0), -1) 
                        cv2.circle(display, (int(self.cursor_x), int(self.cursor_y)), 18, (0, 255, 0), 3)
                    else:
                        cv2.circle(display, (int(self.cursor_x), int(self.cursor_y)), 6, (255, 255, 255), -1) 
                        cv2.circle(display, (int(self.cursor_x), int(self.cursor_y)), 16, (255, 255, 255), 2)

                # FPS and Engine telemetry
                fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
                prev_time = current_time
                self.fps_history.append(fps)
                self.draw_debug_hud(display)

                # 16:10 aspect ratio wrapper
                target_h = int(self.w * (10 / 16))
                canvas = np.zeros((target_h, self.w, 3), dtype=np.uint8)
                y_offset = (target_h - self.h) // 2
                canvas[y_offset:y_offset+self.h, 0:self.w] = display
                
                cv2.imshow(self.window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                elif key == ord('r'): self.reset_game()
                elif key == ord('d'): self.show_debug = not self.show_debug
                elif key == ord('p'): self.show_camera_panel = not self.show_camera_panel

        finally:
            self.inference_running = False
            if self.inference_thread.is_alive():
                self.inference_thread.join()
            self.stream.stop()
            cv2.destroyAllWindows()

    def handle_logic(self):
        cx, cy = self.cursor_x, self.cursor_y
        
        if self.is_pinching:
            # Z-INDEX: Iterate backward so you pick up the top-most component
            if self.grabbed_idx is None:
                for i in range(len(self.components) - 1, -1, -1):
                    comp = self.components[i]
                    if not comp.snapped and (comp.x < cx < comp.x + comp.w) and (comp.y < cy < comp.y + comp.h):
                        self.grabbed_idx = i
                        break
            
            if self.grabbed_idx is not None:
                comp = self.components[self.grabbed_idx]
                comp.target_render_x = cx - (comp.w / 2)
                comp.target_render_y = cy - (comp.h / 2)
        else:
            if self.grabbed_idx is not None:
                comp = self.components[self.grabbed_idx]
                
                c_center_x = comp.target_render_x + (comp.w / 2)
                c_center_y = comp.target_render_y + (comp.h / 2)
                t_center_x = comp.target_x + (comp.w / 2)
                t_center_y = comp.target_y + (comp.h / 2)
                
                if math.hypot(c_center_x - t_center_x, c_center_y - t_center_y) < 60:
                    comp.target_render_x, comp.target_render_y = comp.target_x, comp.target_y
                    comp.snapped = True
                    self.particles.append(Particle(t_center_x, t_center_y))

                    if all(c.snapped for c in self.components):
                        self.game_state = "WIN"
                '''else:
                    # SNAP-BACK: Invalid drop returns the component to the tray
                    comp.target_render_x = comp.start_x
                    comp.target_render_y = comp.start_y'''
                    
                self.grabbed_idx = None

    def reset_game(self):
        for comp in self.components:
            comp.target_render_x, comp.target_render_y = comp.start_x, comp.start_y
            comp.snapped = False
        self.grabbed_idx = None
        self.game_state = "PLAYING"

if __name__ == "__main__":
    game = CircuitBuilderGame()
    game.run()
