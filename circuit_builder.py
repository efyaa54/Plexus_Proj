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
    (9, 10), (10, 11), (11, 12),           # Middle
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

    '''def draw(self, surface, is_grabbed, current_time, debug=False):
        if not self.snapped:
            if is_grabbed:
                # --- GLOWING SOCKET EFFECT ---
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
        
        if is_grabbed:
            shadow_offset = 15
            if self.img is not None and self.img.shape[2] == 4:
                # Create a shape-matching shadow using the image's alpha channel
                shadow_img = np.zeros((self.h, self.w, 4), dtype=np.uint8)
                shadow_img[:, :, :3] = (10, 10, 10)  # Dark shadow color
                shadow_img[:, :, 3] = (self.img[:, :, 3] * 0.5).astype(np.uint8)  # Semi-transparent
                overlay_transparent(surface, shadow_img, self.x + shadow_offset, self.y + shadow_offset)
            else:
                # Fallback to rectangular shadow if no alpha channel exists
                cv2.rectangle(surface, 
                              (int(self.x) + shadow_offset, int(self.y) + shadow_offset), 
                              (int(self.x + self.w) + shadow_offset, int(self.y + self.h) + shadow_offset), 
                              (10, 10, 10), -1)
                              
            cv2.rectangle(surface, (int(self.x)-2, int(self.y)-2), 
                          (int(self.x + self.w)+2, int(self.y + self.h)+2), 
                          (0, 255, 0), 3)

        overlay_transparent(surface, self.img, self.x, self.y)
        
        if debug:
            cv2.rectangle(surface, (int(self.x), int(self.y)), 
                          (int(self.x + self.w), int(self.y + self.h)), (255, 0, 255), 1)'''

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
        
        self.components = []
        self.particles = []
        self.grabbed_idx = None
        self.cursor_x, self.cursor_y = self.w / 2, self.h / 2
        
        self.is_pinching = False
        self.PINCH_GRAB_THRESH = 60
        self.PINCH_DROP_THRESH = 100

        self.show_pip = False  
        
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
        scale = self.h / raw_board.shape[0]
        self.new_w, self.new_h = int(raw_board.shape[1] * scale), int(raw_board.shape[0] * scale)
        self.board_img = cv2.resize(raw_board, (self.new_w, self.new_h))
        self.board_x = (self.w - self.new_w) // 2
        self.board_y = (self.h - self.new_h) // 2
        for comp_data in config["components"]:
            self.components.append(Component(comp_data))

    def draw_debug_hud(self, surface):
        if not self.show_debug: return
        
        hud_w, hud_h = 300, 150
        cv2.rectangle(surface, (10, 10), (10 + hud_w, 10 + hud_h), (0, 0, 0, 150), -1)
        cv2.putText(surface, "[DEBUG TELEMETRY]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        current_fps = self.fps_history[-1] if self.fps_history else 0
        current_inf = self.inf_history[-1] if self.inf_history else 0
        
        cv2.putText(surface, f"Render FPS: {int(current_fps)}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(surface, f"Infer Latency: {int(current_inf)}ms", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(surface, f"Hand Conf: {self.confidence_score:.2f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(surface, f"Pinch State: {self.is_pinching}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        if len(self.fps_history) > 1:
            pts = []
            for i, fps_val in enumerate(self.fps_history):
                x = 20 + int((i / 60) * 280)
                y = 150 - int((min(fps_val, 120) / 120) * 30)
                pts.append((x, y))
            cv2.polylines(surface, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), 1)

    def run(self):
        prev_time = time.perf_counter()
        try:
            while True:
                with self.inference_lock:
                    results = self.latest_result
                
                display = np.full((self.h, self.w, 3), (25, 25, 25), dtype=np.uint8)
                display[self.board_y:self.board_y+self.new_h, self.board_x:self.board_x+self.new_w] = self.board_img

                target_cursor_x, target_cursor_y = self.cursor_x, self.cursor_y
                cursor_active = False

                if results and results.hand_landmarks:
                    cursor_active = True
                    lm = results.hand_landmarks[0]
                    
                    # --- ACTIVE CONTROL AREA MATH ---
                    # Defines a 20% margin on all sides. Hand only needs to move in the center 60% of the camera.
                    margin_x, margin_y = 0.20, 0.20 
                    
                    def map_coord(val, margin, max_val):
                        # Map normalized coordinate [margin, 1-margin] to [0, max_val]
                        mapped = (val - margin) / (1.0 - 2.0 * margin)
                        # Clamp the values so the cursor stops exactly at the screen edge
                        return max(0.0, min(float(max_val), mapped * max_val))
                        
                    t_x = map_coord(lm[4].x, margin_x, self.w)
                    t_y = map_coord(lm[4].y, margin_y, self.h)
                    i_x = map_coord(lm[8].x, margin_x, self.w)
                    i_y = map_coord(lm[8].y, margin_y, self.h)
                    
                    target_cursor_x = (t_x + i_x) / 2
                    target_cursor_y = (t_y + i_y) / 2
                    
                    distance = math.hypot(t_x - i_x, t_y - i_y)
                    
                    if not self.is_pinching and distance < self.PINCH_GRAB_THRESH:
                        self.is_pinching = True
                    elif self.is_pinching and distance > self.PINCH_DROP_THRESH:
                        self.is_pinching = False

                self.cursor_x = lerp(self.cursor_x, target_cursor_x, 0.5)
                self.cursor_y = lerp(self.cursor_y, target_cursor_y, 0.5)

                self.handle_logic()

                current_time = time.perf_counter()
                
                # --- PASS 1: Draw all socket boxes first ---
                for i, comp in enumerate(self.components):
                    comp.draw_socket(display, is_grabbed=(self.grabbed_idx == i), current_time=current_time)

                # --- PASS 2: Update and draw all components on top ---
                for i, comp in enumerate(self.components):
                    comp.update()
                    comp.draw_component(display, is_grabbed=(self.grabbed_idx == i), debug=self.show_debug)

                for p in self.particles[:]:
                    p.update()
                    p.draw(display)
                    if p.life <= 0: self.particles.remove(p)

                if cursor_active:
                    if self.is_pinching:
                        cv2.circle(display, (int(self.cursor_x), int(self.cursor_y)), 12, (0, 255, 0), -1) 
                        cv2.circle(display, (int(self.cursor_x), int(self.cursor_y)), 18, (0, 255, 0), 3)
                    else:
                        cv2.circle(display, (int(self.cursor_x), int(self.cursor_y)), 6, (255, 255, 255), -1) 
                        cv2.circle(display, (int(self.cursor_x), int(self.cursor_y)), 16, (255, 255, 255), 2)

                # --- WEBCAM PICTURE-IN-PICTURE (PiP) FEED ---
                if self.show_pip:
                    success_pip, pip_frame = self.stream.read()
                    if success_pip and pip_frame is not None:
                        pip_frame = cv2.flip(pip_frame, 1)
                        pip_w, pip_h = 320, 180
                        pip_resized = cv2.resize(pip_frame, (pip_w, pip_h))
                        
                        # Draw the Active Control Zone boundary inside the PiP
                        margin_x, margin_y = 0.20, 0.20
                        cv2.rectangle(pip_resized, 
                                     (int(pip_w * margin_x), int(pip_h * margin_y)), 
                                     (int(pip_w * (1.0 - margin_x)), int(pip_h * (1.0 - margin_y))), 
                                     (255, 50, 50), 1) # Subtle Blue border
                        
                        if results and results.hand_landmarks:
                            lms = results.hand_landmarks[0]
                            for start_idx, end_idx in HAND_CONNECTIONS:
                                x1, y1 = int(lms[start_idx].x * pip_w), int(lms[start_idx].y * pip_h)
                                x2, y2 = int(lms[end_idx].x * pip_w), int(lms[end_idx].y * pip_h)
                                cv2.line(pip_resized, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            for lm in lms:
                                cx, cy = int(lm.x * pip_w), int(lm.y * pip_h)
                                cv2.circle(pip_resized, (cx, cy), 4, (255, 0, 0), -1)

                        # Positioned in the bottom right corner with padding
                        pip_x, pip_y = self.w - pip_w - 30, self.h - pip_h - 30
                        draw_rounded_rect(display, (pip_x - 3, pip_y - 3), (pip_x + pip_w + 3, pip_y + pip_h + 3), (100, 120, 150), 2, 8)
                        display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = pip_resized
                        cv2.putText(display, "LIVE HAND TRACKING", (pip_x + 8, pip_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
                prev_time = current_time
                self.fps_history.append(fps)
                self.draw_debug_hud(display)

                target_h = int(self.w * (10 / 16))
                canvas = np.zeros((target_h, self.w, 3), dtype=np.uint8)
                y_offset = (target_h - self.h) // 2
                canvas[y_offset:y_offset+self.h, 0:self.w] = display
                
                cv2.imshow(self.window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                elif key == ord('r'): self.reset_game()
                elif key == ord('d'): self.show_debug = not self.show_debug
                elif key == ord('p'): self.show_pip = not self.show_pip

        finally:
            self.inference_running = False
            if self.inference_thread.is_alive():
                self.inference_thread.join()
            self.stream.stop()
            cv2.destroyAllWindows()

    def handle_logic(self):
        cx, cy = self.cursor_x, self.cursor_y
        
        if self.is_pinching:
            if self.grabbed_idx is None:
                for i, comp in enumerate(self.components):
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
                    
                self.grabbed_idx = None

    def reset_game(self):
        for comp in self.components:
            comp.target_render_x, comp.target_render_y = comp.start_x, comp.start_y
            comp.snapped = False
        self.grabbed_idx = None

if __name__ == "__main__":
    game = CircuitBuilderGame()
    game.run()
