import cv2
import numpy as np
import mediapipe as mp
import time
import math
import threading
import collections
import subprocess
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import json
import os
from datetime import datetime

SCORE_FILE = "player_data.json"

def load_player_data():
    if os.path.exists(SCORE_FILE):
        try:
            with open(SCORE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "active_player" in data:
                    return data
        except json.JSONDecodeError:
            pass
            
    # Default fallback if file doesn't exist or is legacy format
    return {
        "active_player": {"name": "Guest", "score": 0, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "leaderboard": []
    }

def save_player_data(data):
    # Ensure active player is also recorded or updated in the leaderboard history
    active = data["active_player"]
    leaderboard = data.get("leaderboard", [])
    
    # Check if this exact session already exists in leaderboard, update it or append
    found = False
    for entry in leaderboard:
        if entry["name"] == active["name"] and entry["timestamp"] == active["timestamp"]:
            entry["score"] = active["score"]
            found = True
            break
            
    if not found:
        leaderboard.append(active.copy())
        
    data["leaderboard"] = leaderboard
    with open(SCORE_FILE, "w") as f:
        json.dump(data, f, indent=4)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def lerp(current, target, alpha=0.15):
    """Smoothly moves a value towards a target."""
    return current + alpha * (target - current)

def overlay_transparent(background, overlay, x, y):
    """Blends a PNG with an alpha channel onto the background."""
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
class FactoryNavigatorGame:
    def __init__(self):
        self.w, self.h = 1280, 720
        self.window_name = "Plexus Factory Navigator"

        self.player_data = load_player_data()
        self.score_awarded = False

        # --- UNIFIED LAYOUT CONSTANTS ---
        self.game_w, self.game_h = 750, 560
        self.game_x = 50
        self.game_y = (self.h - self.game_h) // 2 + 40
        
        self.panel_x = self.game_x + self.game_w + 40
        self.panel_y = self.game_y
        self.panel_w = (self.w - 40) - self.panel_x
        self.panel_h = self.game_h

        # --- GAME STATE ---
        self.game_state = "START_SCREEN"
        self.distance = 0.0
        self.max_distance = 100.0  
        self.health = 3
        self.hit_flash_timer = 0    
        self.finish_line_obj = None

        # --- DEBUG TELEMETRY ---
        self.debug_mode = False     
        self.fps_history = collections.deque(maxlen=60)
        self.inf_history = collections.deque(maxlen=60)
        self.confidence_score = 0.0

        # --- PLAYER STATE (ROBOT) ---
        self.robot_radius = 70 
        self.robot_speed = 6
        self.target_robot_x = self.game_w / 2
        self.robot_x = self.game_w / 2
        self.robot_y = self.game_h - 100

        # --- BELT & OBSTACLE STATE ---
        self.belt_offset = 0
        self.belt_speed = 2.0
        self.max_belt_speed = 6.0
        self.obstacles = []
        self.spawn_timer = 0
        self.spawn_rate = 50  

        # --- STEERING / LEAN DETECTION ---
        self.LEAN_RATIO_THRESHOLD = 0.35
        self.lean_ratio = 0.0
        self.steering_command = "CENTERED"
        self.steering_color = (120, 220, 255)

        # --- ASSET LOADING ---
        self.load_graphics()

        # --- THREADING & ML ---
        self.setup_window()
        self.stream = WebcamStream(src=0, width=self.w, height=self.h)

        self.latest_result = None
        self.inference_lock = threading.Lock()
        self.inference_running = True

        base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task', delegate=python.BaseOptions.Delegate.CPU)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()

    def load_graphics(self):
        self.robot_img_raw = cv2.imread("pcb.png", cv2.IMREAD_UNCHANGED)
        if self.robot_img_raw is not None:
            diam = self.robot_radius * 2
            self.robot_img = cv2.resize(self.robot_img_raw, (diam, diam))
        else:
            self.robot_img = None
            print("Warning: pcb.png not found. Using fallback shapes.")

        self.hazard_img_raw = cv2.imread("hazard.png", cv2.IMREAD_UNCHANGED)
        if self.hazard_img_raw is None:
            print("Warning: hazard.png not found. Using fallback shapes.")

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
            
            # Ensure timestamp strictly increases for video mode
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
            except Exception:
                pass

            time.sleep(0.001)

    def reset_game(self):
        self.obstacles.clear() 
        self.belt_speed = 2.0 
        self.robot_speed = 6
        self.spawn_rate = 50
        self.distance = 0.0 
        self.health = 3
        self.target_robot_x = self.game_w / 2 
        self.robot_x = self.game_w / 2
        self.finish_line_obj = None
        self.game_state = "PLAYING"
        self.score_awarded = False

    def draw_header(self, display):
        """Draws the top banner UI to match Circuit Builder style."""
        draw_rounded_rect(display, (50, 15), (self.w - 50, 70), (30, 38, 52), cv2.FILLED, 10)
        draw_rounded_rect(display, (50, 15), (self.w - 50, 70), (60, 80, 110), 1, 10)
        
        cv2.putText(display, "PLEXUS FACTORY NAVIGATOR", (75, 52), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)

        active = self.player_data["active_player"]
        player_text = f"PLAYER: {active['name']}  |  SCORE: {active['score']}"
        cv2.putText(display, player_text, (self.w - 800, 52), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 128), 1)
        
        # OVEN PROGRESS BAR
        bar_w = 400
        bar_x1 = self.w - 50 - bar_w - 20
        bar_y1 = 28
        bar_x2 = bar_x1 + bar_w
        bar_y2 = 58
        
        draw_rounded_rect(display, (bar_x1, bar_y1), (bar_x2, bar_y2), (20, 24, 33), cv2.FILLED, 8)
        filled_w = int((bar_x2 - bar_x1) * (self.distance / self.max_distance))
        if filled_w > 0:
            draw_rounded_rect(display, (bar_x1, bar_y1), (bar_x1 + filled_w, bar_y2), (0, 180, 255), cv2.FILLED, 8)
        draw_rounded_rect(display, (bar_x1, bar_y1), (bar_x2, bar_y2), (90, 110, 140), 1, 8)
        
        cv2.putText(display, f"OVEN PROGRESS: {int(self.distance)}m / {int(self.max_distance)}m", 
                    (bar_x1 + 35, bar_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_operator_panel(self, display, results):
        """Draws the right-side dashboard containing the HP, Lean Gauge, and PiP Feed."""
        panel_x, panel_y, panel_w, panel_h = self.panel_x, self.panel_y, self.panel_w, self.panel_h

        draw_rounded_rect(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 38, 52), cv2.FILLED, 15)
        draw_rounded_rect(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (60, 80, 110), 2, 15)

        text_x = panel_x + 30
        current_y = panel_y + 45
        cv2.putText(display, "OPERATOR DASHBOARD", (text_x, current_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        current_y += 40
        card_w = panel_w - 60

        # --- HP INDICATOR ---
        cv2.putText(display, "ROBOT INTEGRITY", (text_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        current_y += 15
        for h_idx in range(3):
            color = (0, 255, 128) if h_idx < self.health else (60, 70, 90)
            hx = text_x + (h_idx * 55)
            draw_rounded_rect(display, (hx, current_y), (hx + 45, current_y + 25), color, cv2.FILLED, 6)
        
        # --- LEAN STEERING GAUGE ---
        current_y += 65
        cv2.putText(display, "LEAN STEERING GAUGE", (text_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        current_y += 15
        
        gauge_h = 30
        cv2.rectangle(display, (text_x, current_y), (text_x + card_w, current_y + gauge_h), (15, 20, 30), cv2.FILLED)
        cv2.rectangle(display, (text_x, current_y), (text_x + card_w, current_y + gauge_h), (90, 110, 140), 1)

        center_tick = text_x + card_w // 2
        max_range = 1.0  
        thresh_offset_px = int((self.LEAN_RATIO_THRESHOLD / max_range) * (card_w / 2))
        
        # Threshold lines
        cv2.line(display, (center_tick - thresh_offset_px, current_y - 2), (center_tick - thresh_offset_px, current_y + gauge_h + 2), (255, 200, 0), 2)
        cv2.line(display, (center_tick + thresh_offset_px, current_y - 2), (center_tick + thresh_offset_px, current_y + gauge_h + 2), (255, 200, 0), 2)

        # Indicator dot
        clamped_ratio = max(-max_range, min(max_range, self.lean_ratio))
        indicator_offset = int((clamped_ratio / max_range) * (card_w / 2))
        cv2.circle(display, (center_tick + indicator_offset, current_y + gauge_h // 2), 7, self.steering_color, cv2.FILLED)

        status_msg = f"STATUS: {self.steering_command}"
        cv2.putText(display, status_msg, (text_x, current_y + gauge_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.steering_color, 1, cv2.LINE_AA)

        # --- LIVE CAMERA FEED ---
        pip_h = min(int(card_w * (9 / 16)), 240)
        pip_w = int(pip_h * (16 / 9))
        pip_x = text_x + (card_w - pip_w) // 2
        panel_bottom = panel_y + panel_h - 20
        pip_y = panel_bottom - pip_h

        cv2.putText(display, "LIVE CAMERA FEED", (pip_x, pip_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        success_pip, pip_frame = self.stream.read()
        if success_pip and pip_frame is not None:
            pip_resized = cv2.resize(cv2.flip(pip_frame, 1), (pip_w, pip_h))
            
            pose_visible = bool(results and results.pose_landmarks)
            if pose_visible:
                lms = results.pose_landmarks[0]
                lx_p, ly_p = int(lms[11].x * pip_w), int(lms[11].y * pip_h)
                rx_p, ry_p = int(lms[12].x * pip_w), int(lms[12].y * pip_h)
                lean_color = (0, 255, 128) if abs(self.lean_ratio) > self.LEAN_RATIO_THRESHOLD else (0, 220, 255)
                cv2.line(pip_resized, (lx_p, ly_p), (rx_p, ry_p), lean_color, 2)
                cv2.circle(pip_resized, (lx_p, ly_p), 6, lean_color, -1)
                cv2.circle(pip_resized, (rx_p, ry_p), 6, lean_color, -1)

            draw_rounded_rect(display, (pip_x - 3, pip_y - 3), (pip_x + pip_w + 3, pip_y + pip_h + 3), (100, 120, 150), 2, 8)
            display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = pip_resized

            if not pose_visible:
                warn = display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w].copy()
                cv2.rectangle(warn, (0, 0), (pip_w, pip_h), (0, 0, 0), cv2.FILLED)
                cv2.addWeighted(warn, 0.55, display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w], 0.45, 0,
                                display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w])
                cv2.putText(display, "NO PLAYER DETECTED", (pip_x + 20, pip_y + pip_h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 100, 255), 2)

    def draw_debug_hud(self, surface, smoothed_fps):
        if not self.debug_mode: return
        
        hud_w, hud_h = 300, 200
        debug_overlay = surface.copy()
        draw_rounded_rect(debug_overlay, (20, 120), (20 + hud_w, 120 + hud_h), (15, 20, 30), cv2.FILLED, 10)
        cv2.addWeighted(debug_overlay, 0.75, surface, 0.25, 0, surface)
        draw_rounded_rect(surface, (20, 120), (20 + hud_w, 120 + hud_h), (0, 180, 255), 1, 10)

        current_inf = self.inf_history[-1] if self.inf_history else 0
        
        cv2.putText(surface, "[DEBUG TELEMETRY]", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
        cv2.putText(surface, f"Render FPS: {int(smoothed_fps)}", (35, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)
        cv2.putText(surface, f"Infer Latency: {int(current_inf)}ms", (35, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(surface, f"Active Obstacles: {len(self.obstacles)}", (35, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(surface, f"Belt Speed: {self.belt_speed:.2f}", (35, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(surface, f"Lean Ratio: {self.lean_ratio:.2f}", (35, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(surface, f"Inference Thread Active: {self.inference_thread.is_alive()}", (35, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128) if self.inference_thread.is_alive() else (0, 0, 255), 1)

    def run(self):
        prev_time = time.time()
        smoothed_fps = 0.0

        try:
            while True:
                with self.inference_lock:
                    results = self.latest_result

                # 1. Base display background
                display = np.full((self.h, self.w, 3), (25, 25, 25), dtype=np.uint8)
                
                # 2. Local canvas just for the game area
                game_canvas = np.full((self.game_h, self.game_w, 3), (20, 24, 33), dtype=np.uint8)

                self.steering_command = "CENTERED"
                self.steering_color = (120, 220, 255)

                key = cv2.waitKey(1) & 0xFF
                if key == 27: 
                    break
                elif key == ord('d') or key == ord('D'):
                    self.debug_mode = not self.debug_mode

                if self.game_state == "START_SCREEN":
                    if key == ord(' ') or key == 13: 
                        self.reset_game()

                elif self.game_state == "PLAYING":
                    if self.hit_flash_timer > 0:
                        self.hit_flash_timer -= 1

                    self.distance += 0.05
                    if self.distance >= self.max_distance:
                        self.distance = self.max_distance
                        self.game_state = "WIN"

                    if int(self.distance) % 20 == 0 and int(self.distance) > 0:
                        if self.belt_speed < self.max_belt_speed:
                            self.belt_speed = min(self.max_belt_speed, self.belt_speed + 0.002)

                    self.belt_offset += self.belt_speed
                    if self.belt_offset > 100: 
                        self.belt_offset = 0

                    frames_to_reach_robot = (self.robot_y + 100) / self.belt_speed
                    distance_covered_in_frames = frames_to_reach_robot * 0.05
                    spawn_distance = self.max_distance - distance_covered_in_frames

                    if self.distance >= spawn_distance and self.finish_line_obj is None:
                        self.finish_line_obj = {"y": -100.0}

                    if self.finish_line_obj is not None:
                        self.finish_line_obj["y"] += self.belt_speed

                    if self.distance < spawn_distance:
                        self.spawn_timer += 1
                        if self.spawn_timer >= self.spawn_rate:
                            self.spawn_timer = 0
                            obs_x = random.randint(90, self.game_w - 90)
                            obs_radius = random.randint(30, 50) 

                            scaled_hazard = None
                            if self.hazard_img_raw is not None:
                                diam = obs_radius * 2
                                scaled_hazard = cv2.resize(self.hazard_img_raw, (diam, diam))

                            self.obstacles.append({
                                "x": obs_x, 
                                "y": -100, 
                                "r": obs_radius,
                                "img": scaled_hazard
                            })

                    for obs in self.obstacles[:]:
                        obs["y"] += self.belt_speed
                        if obs["y"] > self.game_h + 100:
                            self.obstacles.remove(obs)
                        elif math.hypot(self.robot_x - obs["x"], self.robot_y - obs["y"]) < (self.robot_radius + obs["r"]) * 0.6: 
                            if self.hit_flash_timer == 0: 
                                self.health -= 1
                                self.hit_flash_timer = 45 
                                self.belt_speed = max(2.0, self.belt_speed - 1.0)
                                self.obstacles.remove(obs)

                                if self.health <= 0:
                                    self.game_state = "GAME_OVER"
                                    break

                    if results and results.pose_landmarks:
                        landmarks = results.pose_landmarks[0]
                        left_shoulder = landmarks[11]
                        right_shoulder = landmarks[12]

                        shoulder_width = max(abs(left_shoulder.x - right_shoulder.x), 1e-4)
                        raw_lean_diff = left_shoulder.y - right_shoulder.y
                        self.lean_ratio = raw_lean_diff / shoulder_width

                        if self.lean_ratio > self.LEAN_RATIO_THRESHOLD:
                            self.steering_command = "TURNING RIGHT"
                            self.steering_color = (0, 255, 128) 
                            self.target_robot_x += self.robot_speed 
                        elif self.lean_ratio < -self.LEAN_RATIO_THRESHOLD:
                            self.steering_command = "TURNING LEFT"
                            self.steering_color = (60, 120, 255) 
                            self.target_robot_x -= self.robot_speed 

                    self.target_robot_x = max(self.robot_radius, min(self.game_w - self.robot_radius, self.target_robot_x))
                    self.robot_x = lerp(self.robot_x, self.target_robot_x, alpha=0.15)

                elif self.game_state in ["GAME_OVER", "WIN"]:
                    if key == ord('r') or key == ord('R'):
                        self.reset_game()

                # --- RENDER LOCAL GAME CANVAS ---
                for i in range(-100, self.game_h, 100):
                    line_y = i + self.belt_offset
                    cv2.line(game_canvas, (0, int(line_y)), (self.game_w, int(line_y)), (38, 48, 64), 3, cv2.LINE_AA)

                cv2.line(game_canvas, (20, 0), (20, self.game_h), (0, 180, 255), 4)
                cv2.line(game_canvas, (self.game_w - 20, 0), (self.game_w - 20, self.game_h), (0, 180, 255), 4)

                if self.finish_line_obj is not None:
                    fy = int(self.finish_line_obj["y"])
                    if -100 <= fy <= self.game_h + 100:
                        cv2.rectangle(game_canvas, (40, fy - 12), (self.game_w - 40, fy + 12), (255, 255, 255), cv2.FILLED)
                        cv2.rectangle(game_canvas, (40, fy - 12), (self.game_w - 40, fy + 12), (0, 215, 255), 2)
                        cv2.putText(game_canvas, "REFLOW OVEN EXIT / FINISH LINE", (self.game_w // 2 - 200, fy + 6), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 24, 33), 2)

                for obs in self.obstacles:
                    if obs["img"] is not None:
                        game_canvas = overlay_transparent(game_canvas, obs["img"], obs["x"] - obs["r"], obs["y"] - obs["r"])
                    else:
                        cv2.circle(game_canvas, (obs["x"], int(obs["y"])), obs["r"], (0, 100, 255), cv2.FILLED)

                rx = int(self.robot_x)
                if self.hit_flash_timer > 0 and (self.hit_flash_timer // 5) % 2 == 0:
                    if self.robot_img is not None:
                        red_robot = self.robot_img.copy()
                        red_robot[:, :, 0] = 0 
                        red_robot[:, :, 1] = 0 
                        game_canvas = overlay_transparent(game_canvas, red_robot, rx - self.robot_radius, self.robot_y - self.robot_radius)
                    else:
                        cv2.circle(game_canvas, (rx, self.robot_y), self.robot_radius, (0, 0, 255), cv2.FILLED)
                else:
                    if self.robot_img is not None:
                        game_canvas = overlay_transparent(game_canvas, self.robot_img, rx - self.robot_radius, self.robot_y - self.robot_radius)
                    else:
                        cv2.circle(game_canvas, (rx, self.robot_y), self.robot_radius, (28, 41, 218), cv2.FILLED)

                # Map local game_canvas to the main display
                display[self.game_y:self.game_y+self.game_h, self.game_x:self.game_x+self.game_w] = game_canvas
                # Draw sleek border around the game area
                draw_rounded_rect(display, (self.game_x, self.game_y), (self.game_x + self.game_w, self.game_y + self.game_h), (60, 80, 110), 2, 10)

                # --- DRAW UI PANELS ---
                self.draw_header(display)
                self.draw_operator_panel(display, results)

                # --- OVERLAYS ---
                if self.game_state == "START_SCREEN":
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (self.w, self.h), (10, 12, 18), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)

                    draw_rounded_rect(display, (self.w//2 - 460, self.h//2 - 200), (self.w//2 + 460, self.h//2 + 180), (45, 55, 75), cv2.FILLED, 20)
                    draw_rounded_rect(display, (self.w//2 - 460, self.h//2 - 200), (self.w//2 + 460, self.h//2 + 180), (0, 180, 255), 2, 20)

                    cv2.putText(display, "PLEXUS FACTORY NAVIGATOR", (self.w//2 - 320, self.h//2 - 130), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 215, 255), 2)
                    cv2.putText(display, "GOAL: Guide your PCB safely through the reflow oven to 100m!", (self.w//2 - 390, self.h//2 - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 1)
                    cv2.putText(display, "CONTROLS: Lean Left or Right with your shoulders to steer.", (self.w//2 - 380, self.h//2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 200, 220), 1)

                    is_tracking = results is not None and results.pose_landmarks is not None and len(results.pose_landmarks) > 0
                    track_status = "PLAYER DETECTED - READY!" if is_tracking else "SEARCHING FOR PLAYER (STAND BACK)..."
                    track_color = (0, 255, 128) if is_tracking else (0, 165, 255)
                    cv2.putText(display, track_status, (self.w//2 - 210, self.h//2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 1)

                    cv2.putText(display, "Press [SPACEBAR] or [ENTER] to Start", (self.w//2 - 220, self.h//2 + 115), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)

                elif self.game_state == "GAME_OVER":
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (self.w, self.h), (10, 10, 15), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

                    draw_rounded_rect(display, (self.w//2 - 360, self.h//2 - 160), (self.w//2 + 360, self.h//2 + 140), (40, 25, 30), cv2.FILLED, 16)
                    draw_rounded_rect(display, (self.w//2 - 360, self.h//2 - 160), (self.w//2 + 360, self.h//2 + 140), (0, 0, 255), 2, 16)

                    cv2.putText(display, "SYSTEM FAILURE", (self.w//2 - 210, self.h//2 - 80), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
                    cv2.putText(display, f"FINAL DISTANCE: {int(self.distance)}m", (self.w//2 - 180, self.h//2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                    cv2.putText(display, "Press 'R' to Retry  |  Press 'ESC' to Quit", (self.w//2 - 230, self.h//2 + 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 170, 170), 1)

                elif self.game_state == "WIN":
                    if not self.score_awarded:
                        self.player_data["active_player"]["score"] += 1
                        save_player_data(self.player_data)
                        self.score_awarded = True
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (self.w, self.h), (10, 15, 20), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

                    draw_rounded_rect(display, (self.w//2 - 380, self.h//2 - 160), (self.w//2 + 380, self.h//2 + 140), (25, 45, 35), cv2.FILLED, 16)
                    draw_rounded_rect(display, (self.w//2 - 380, self.h//2 - 160), (self.w//2 + 380, self.h//2 + 140), (0, 255, 128), 2, 16)

                    cv2.putText(display, "REFLOW COMPLETE!", (self.w//2 - 240, self.h//2 - 80), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 128), 2)
                    cv2.putText(display, "Successfully reached 100 meters!", (self.w//2 - 230, self.h//2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(display, "Press 'R' to Play Again  |  Press 'ESC' to Quit", (self.w//2 - 250, self.h//2 + 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 170, 170), 1)

                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time
                instant_fps = 1 / dt if dt > 0 else 0
                smoothed_fps = lerp(smoothed_fps, instant_fps, 0.1)

                self.draw_debug_hud(display, smoothed_fps)

                target_h = int(self.w * (10 / 16))
                canvas = np.zeros((target_h, self.w, 3), dtype=np.uint8)
                y_offset = (target_h - self.h) // 2
                canvas[y_offset:y_offset+self.h, 0:self.w] = display

                cv2.imshow(self.window_name, canvas)

        finally:
            self.inference_running = False
            if self.inference_thread.is_alive():
                self.inference_thread.join()
            self.stream.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    game = FactoryNavigatorGame()
    game.run()
