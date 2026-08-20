import cv2
import numpy as np
import mediapipe as mp
import time
import math
import threading
import collections
import subprocess
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def lerp(current, target, alpha=0.3):
    return current + alpha * (target - current)

def draw_rounded_rect(img, pt1, pt2, color, thickness=1, radius=15):
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
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret, self.frame = ret, frame
            except Exception as e:
                pass

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

# ==========================================
# GAME MANAGER
# ==========================================
class DefectDetectiveGame:
    def __init__(self):
        self.w, self.h = 1280, 720
        self.window_name = "Plexus Defect Detective"
        self.lens_radius = 160

        # --- GAME STATE ---
        self.game_state = "PLAYING"
        self.cursor_x, self.cursor_y = float(self.w / 2), float(self.h / 2)
        self.pinch_cooldown = 0
        self.is_pinching = False
        self.PINCH_GRAB_THRESH = 0.05
        self.PINCH_DROP_THRESH = 0.07

        # --- DEBUG & UI TOGGLES ---
        self.show_debug = False
        self.show_pip = True
        self.fps_history = collections.deque(maxlen=60)
        self.inf_history = collections.deque(maxlen=60)
        self.confidence_score = 0.0

        # --- DEFECTS SETUP ---
        self.raw_defects = [
            {"x": 239, "y": 716, "r": 35, "found": False},
            {"x": 479, "y": 211, "r": 35, "found": False},
            {"x": 512, "y": 727, "r": 35, "found": False},
            {"x": 738, "y": 133, "r": 35, "found": False},
            {"x": 793, "y": 474, "r": 35, "found": False},
            {"x": 1108, "y": 527, "r": 35, "found": False}
        ]
        self.defects = []

        # --- ASSET LOADING ---
        self.load_graphics()

        # --- THREADING & ML ---
        self.setup_window()
        self.stream = WebcamStream(src=0, width=self.w, height=self.h)

        self.latest_result = None
        self.inference_lock = threading.Lock()
        self.inference_running = True

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task', delegate=python.BaseOptions.Delegate.CPU)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()

    def load_graphics(self):
        raw_good = cv2.imread("board_good.png")
        raw_bad = cv2.imread("board_defect.png")

        if raw_good is not None and raw_bad is not None:
            orig_h, orig_w = raw_good.shape[:2]
            scale = self.h / orig_h
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            
            self.normal_board = cv2.resize(raw_good, (new_w, new_h))
            self.xray_board = cv2.resize(raw_bad, (new_w, new_h))
            
            self.board_x = (self.w - new_w) // 2
            self.board_y = 0
            
            self.bg_normal = np.full((self.h, self.w, 3), (20, 24, 33), dtype=np.uint8)
            self.bg_xray = np.full((self.h, self.w, 3), (20, 24, 33), dtype=np.uint8)
            
            self.bg_normal[self.board_y:self.board_y+new_h, self.board_x:self.board_x+new_w] = self.normal_board
            self.bg_xray[self.board_y:self.board_y+new_h, self.board_x:self.board_x+new_w] = self.xray_board

            # Automatically map the raw image coordinates to the scaled window canvas
            self.defects = []
            for d in self.raw_defects:
                scaled_x = int(d["x"] * scale) + self.board_x
                scaled_y = int(d["y"] * scale) + self.board_y
                scaled_r = int(d["r"] * scale)
                self.defects.append({"x": scaled_x, "y": scaled_y, "r": max(25, scaled_r), "found": False})

        else:
            print("Warning: board_good.png / board_bad.png not found. Using synthetic patterns.")
            self.bg_normal = np.full((self.h, self.w, 3), (30, 80, 20), dtype=np.uint8)
            self.bg_xray = np.full((self.h, self.w, 3), (30, 20, 20), dtype=np.uint8)
            self.defects = [{"x": d["x"], "y": d["y"], "r": d["r"], "found": False} for d in self.raw_defects]
            
            for i in range(12):
                cx, cy = (i * 120 + 80) % self.w, (i * 150 + 100) % self.h
                cv2.rectangle(self.bg_normal, (cx-40, cy-40), (cx+40, cy+40), (150, 150, 150), -1)
                cv2.rectangle(self.bg_xray, (cx-40, cy-40), (cx+40, cy+40), (80, 80, 80), 2)

            for d in self.defects:
                cv2.circle(self.bg_normal, (d["x"], d["y"]), d["r"], (180, 180, 180), -1)
                cv2.circle(self.bg_xray, (d["x"], d["y"]), d["r"], (0, 0, 255), -1)

    def setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def inference_worker(self):
        start_time = time.time()
        last_timestamp = -1

        while self.inference_running:
            success, frame = self.stream.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            timestamp_ms = int((time.time() - start_time) * 1000)
            if timestamp_ms <= last_timestamp:
                timestamp_ms = last_timestamp + 1
            last_timestamp = timestamp_ms

            try:
                start_t = time.perf_counter()
                result = self.detector.detect_for_video(mp_image, timestamp_ms)
                inf_time = (time.perf_counter() - start_t) * 1000

                with self.inference_lock:
                    self.latest_result = result
                    self.inf_history.append(inf_time)
                    if result.hand_landmarks:
                        self.confidence_score = getattr(result.handedness[0][0], 'score', 1.0)
                    else:
                        self.confidence_score = 0.0
            except Exception:
                pass

            time.sleep(0.001)

    def reset_game(self):
        for d in self.defects:
            d["found"] = False
        self.game_state = "PLAYING"
        self.pinch_cooldown = 0

    def draw_debug_hud(self, surface, smoothed_fps):
        if not self.show_debug: return
        
        hud_w, hud_h = 300, 150
        debug_overlay = surface.copy()
        draw_rounded_rect(debug_overlay, (10, 10), (10 + hud_w, 10 + hud_h), (15, 20, 30), cv2.FILLED, 10)
        cv2.addWeighted(debug_overlay, 0.75, surface, 0.25, 0, surface)
        draw_rounded_rect(surface, (10, 10), (10 + hud_w, 10 + hud_h), (0, 180, 255), 1, 10)

        current_inf = self.inf_history[-1] if self.inf_history else 0
        
        cv2.putText(surface, "[DEBUG TELEMETRY]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
        cv2.putText(surface, f"Render FPS: {int(smoothed_fps)}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)
        cv2.putText(surface, f"Infer Latency: {int(current_inf)}ms", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(surface, f"Hand Conf: {self.confidence_score:.2f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(surface, f"Pinch State: {self.is_pinching}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    def run(self):
        prev_time = time.time()
        smoothed_fps = 0.0

        try:
            while True:
                with self.inference_lock:
                    results = self.latest_result

                display = self.bg_normal.copy()
                target_cursor_x, target_cursor_y = self.cursor_x, self.cursor_y
                cursor_active = False

                if results and results.hand_landmarks:
                    cursor_active = True
                    lm = results.hand_landmarks[0]
                    
                    # 1. Calculate pinch distance in raw normalized camera space (0.0 to 1.0)
                    t_x, t_y = lm[4].x, lm[4].y
                    i_x, i_y = lm[8].x, lm[8].y
                    pinch_dist = math.hypot(t_x - i_x, t_y - i_y)
                    
                    # Generous thresholds for instant grabbing
                    if not self.is_pinching and pinch_dist < 0.08:
                        self.is_pinching = True
                    elif self.is_pinching and pinch_dist > 0.12:
                        self.is_pinching = False

                    # 2. Map the knuckle (Landmark 9) through the Active Control Zone for the cursor
                    margin_x, margin_y = 0.20, 0.20
                    def map_coord(val, margin, max_val):
                        mapped = (val - margin) / (1.0 - 2.0 * margin)
                        return max(0.0, min(float(max_val), mapped * max_val))

                    target_cursor_x = map_coord(lm[9].x, margin_x, self.w)
                    target_cursor_y = map_coord(lm[9].y, margin_y, self.h)

                    pinch_dist = math.hypot(t_x - i_x, t_y - i_y)
                    if not self.is_pinching and pinch_dist < self.PINCH_GRAB_THRESH:
                        self.is_pinching = True
                    elif self.is_pinching and pinch_dist > self.PINCH_DROP_THRESH:
                        self.is_pinching = False

                self.cursor_x = lerp(self.cursor_x, target_cursor_x, 0.4)
                self.cursor_y = lerp(self.cursor_y, target_cursor_y, 0.4)
                cx, cy = int(self.cursor_x), int(self.cursor_y)

                # --- RENDER X-RAY LENS EFFECT ---
                mask = np.zeros((self.h, self.w), dtype=np.uint8)
                cv2.circle(mask, (cx, cy), self.lens_radius, 255, -1)
                
                cv2.copyTo(self.bg_xray, mask, display)

                lens_roi = display[max(0, cy-self.lens_radius):min(self.h, cy+self.lens_radius), 
                                   max(0, cx-self.lens_radius):min(self.w, cx+self.lens_radius)]
                
                if lens_roi.size > 0:
                    red_tint = np.full_like(lens_roi, (0, 0, 180), dtype=np.uint8) # BGR Red
                    
                    local_mask = np.zeros((lens_roi.shape[0], lens_roi.shape[1]), dtype=np.uint8)
                    local_cx, local_cy = cx - max(0, cx-self.lens_radius), cy - max(0, cy-self.lens_radius)
                    cv2.circle(local_mask, (local_cx, local_cy), self.lens_radius, 255, -1)
                    
                    blended_roi = cv2.addWeighted(lens_roi, 0.65, red_tint, 0.35, 0)
                    np.copyto(lens_roi, blended_roi, where=local_mask[..., None] == 255)

                lens_color = (0, 0, 255) if self.is_pinching else (0, 80, 255)
                cv2.circle(display, (cx, cy), self.lens_radius, lens_color, 4)
                cv2.circle(display, (cx, cy), 5, lens_color, -1)

                # --- GAME LOGIC ---
                if self.game_state == "PLAYING":
                    if self.pinch_cooldown > 0:
                        self.pinch_cooldown -= 1

                    if self.is_pinching and self.pinch_cooldown == 0:
                        self.pinch_cooldown = 25
                        for d in self.defects:
                            if not d["found"]:
                                if math.hypot(cx - d["x"], cy - d["y"]) < self.lens_radius:
                                    d["found"] = True

                    defects_found = sum(1 for d in self.defects if d["found"])

                    for d in self.defects:
                        if d["found"]:
                            cv2.circle(display, (d["x"], d["y"]), d["r"] + 10, (0, 255, 0), 4)
                            cv2.putText(display, "FOUND", (d["x"] - 35, d["y"] - 45), 
                                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

                    draw_rounded_rect(display, (30, 20), (450, 110), (30, 38, 52), cv2.FILLED, 12)
                    draw_rounded_rect(display, (30, 20), (450, 110), (60, 80, 110), 1, 12)
                    cv2.putText(display, f"DEFECTS FOUND: {defects_found}/{len(self.defects)}", (50, 55), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(display, "Scan with lens. Pinch to flag defects.", (50, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 240), 1)

                    if defects_found == len(self.defects):
                        self.game_state = "WIN"

                elif self.game_state == "WIN":
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (self.w, self.h), (10, 20, 15), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

                    draw_rounded_rect(display, (self.w//2 - 380, self.h//2 - 160), (self.w//2 + 380, self.h//2 + 140), (25, 45, 35), cv2.FILLED, 16)
                    draw_rounded_rect(display, (self.w//2 - 380, self.h//2 - 160), (self.w//2 + 380, self.h//2 + 140), (0, 255, 128), 2, 16)

                    cv2.putText(display, "INSPECTION COMPLETE!", (self.w//2 - 260, self.h//2 - 80), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 128), 2)
                    cv2.putText(display, "All circuit defects successfully isolated.", (self.w//2 - 270, self.h//2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1)
                    cv2.putText(display, "Press 'R' to Play Again  |  Press 'ESC' to Exit", (self.w//2 - 240, self.h//2 + 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 170, 170), 1)

                # --- WEBCAM PICTURE-IN-PICTURE (PiP) FEED ---
                if self.show_pip:
                    success_pip, pip_frame = self.stream.read()
                    if success_pip and pip_frame is not None:
                        pip_frame = cv2.flip(pip_frame, 1)
                        pip_w, pip_h = 320, 180
                        pip_resized = cv2.resize(pip_frame, (pip_w, pip_h))

                        margin_x, margin_y = 0.20, 0.20
                        cv2.rectangle(pip_resized, 
                                     (int(pip_w * margin_x), int(pip_h * margin_y)), 
                                     (int(pip_w * (1.0 - margin_x)), int(pip_h * (1.0 - margin_y))), 
                                     (255, 50, 50), 1)

                        if results and results.hand_landmarks:
                            lms = results.hand_landmarks[0]
                            for start_idx, end_idx in HAND_CONNECTIONS:
                                x1, y1 = int(lms[start_idx].x * pip_w), int(lms[start_idx].y * pip_h)
                                x2, y2 = int(lms[end_idx].x * pip_w), int(lms[end_idx].y * pip_h)
                                cv2.line(pip_resized, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            for lm in lms:
                                lx, ly = int(lm.x * pip_w), int(lm.y * pip_h)
                                cv2.circle(pip_resized, (lx, ly), 4, (0, 215, 255), -1)

                        pip_x, pip_y = self.w - pip_w - 30, self.h - pip_h - 30
                        draw_rounded_rect(display, (pip_x - 3, pip_y - 3), (pip_x + pip_w + 3, pip_y + pip_h + 3), (100, 120, 150), 2, 8)
                        display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = pip_resized
                        cv2.putText(display, "LIVE HAND TRACKING", (pip_x + 8, pip_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # --- FPS & DEBUG HUD ---
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time
                instant_fps = 1 / dt if dt > 0 else 0
                smoothed_fps = lerp(smoothed_fps, instant_fps, 0.1)
                self.fps_history.append(instant_fps)

                self.draw_debug_hud(display, smoothed_fps)

                target_h = int(self.w * (10 / 16))
                canvas = np.zeros((target_h, self.w, 3), dtype=np.uint8)
                y_offset = (target_h - self.h) // 2
                canvas[y_offset:y_offset+self.h, 0:self.w] = display

                cv2.imshow(self.window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    #subprocess.Popen(["python", "arcade_hub_flet.py"])
                    break
                elif key == ord('r') or key == ord('R'):
                    self.reset_game()
                elif key == ord('d') or key == ord('D'):
                    self.show_debug = not self.show_debug
                elif key == ord('p') or key == ord('P'):
                    self.show_pip = not self.show_pip

        finally:
            self.inference_running = False
            if self.inference_thread.is_alive():
                self.inference_thread.join()
            self.stream.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    game = DefectDetectiveGame()
    game.run()
