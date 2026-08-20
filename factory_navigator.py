import cv2
import numpy as np
import mediapipe as mp
import time
import math
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

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
    
    # FIX: Prevent geometry inversion when the progress bar is nearly empty
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
    """Runs the camera on a dedicated background thread to prevent UI freezing."""
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
                print(f"Camera read error: {e}")

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

        # --- GAME STATE ---
        self.game_state = "START_SCREEN"
        self.distance = 0.0
        self.max_distance = 100.0  
        self.health = 3
        self.hit_flash_timer = 0    
        self.debug_mode = False     
        self.finish_line_obj = None

        # --- PLAYER STATE (ROBOT) ---
        self.robot_radius = 120 
        self.robot_speed = 6
        self.target_robot_x = self.w / 2
        self.robot_x = self.w / 2
        self.robot_y = 600

        # --- BELT & OBSTACLE STATE ---
        self.belt_offset = 0
        self.belt_speed = 2.0
        self.max_belt_speed = 6.0
        self.obstacles = []
        self.spawn_timer = 0
        self.spawn_rate = 50  

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
        while self.inference_running:
            success, frame = self.stream.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1) 
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            try:
                result = self.detector.detect_for_video(mp_image, timestamp_ms)
                with self.inference_lock:
                    self.latest_result = result
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
        self.target_robot_x = self.w / 2 
        self.robot_x = self.w / 2
        self.finish_line_obj = None
        self.game_state = "PLAYING"

    def run(self):
        prev_time = time.time()
        smoothed_fps = 0.0
        import random

        try:
            while True:
                with self.inference_lock:
                    results = self.latest_result

                img = np.full((self.h, self.w, 3), (20, 24, 33), dtype=np.uint8)

                steering_command = "CENTERED"
                steering_color = (120, 220, 255)
                shoulder_lean_diff = 0.0

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

                    # FIX: Dynamic Finish Line Calculation
                    # Calculate exactly when to spawn the line so it hits robot_y perfectly at 100m
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
                            obs_x = random.randint(150, self.w - 150)
                            obs_radius = random.randint(45, 75) 

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
                        if obs["y"] > self.h + 100:
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
                        left_shoulder_y = landmarks[11].y
                        right_shoulder_y = landmarks[12].y
                        lean_threshold = 0.05 
                        shoulder_lean_diff = left_shoulder_y - right_shoulder_y

                        if left_shoulder_y > right_shoulder_y + lean_threshold:
                            steering_command = "TURNING RIGHT"
                            steering_color = (0, 255, 128) 
                            self.target_robot_x += self.robot_speed 
                        elif right_shoulder_y > left_shoulder_y + lean_threshold:
                            steering_command = "TURNING LEFT"
                            steering_color = (60, 120, 255) 
                            self.target_robot_x -= self.robot_speed 

                    self.target_robot_x = max(self.robot_radius, min(self.w - self.robot_radius, self.target_robot_x))
                    self.robot_x = lerp(self.robot_x, self.target_robot_x, alpha=0.15)

                elif self.game_state in ["GAME_OVER", "WIN"]:
                    if key == ord('r') or key == ord('R'):
                        self.reset_game()

                # --- RENDERING ---
                for i in range(-100, self.h, 100):
                    line_y = i + self.belt_offset
                    cv2.line(img, (0, int(line_y)), (self.w, int(line_y)), (38, 48, 64), 3, cv2.LINE_AA)

                cv2.line(img, (20, 0), (20, self.h), (0, 180, 255), 4)
                cv2.line(img, (self.w - 20, 0), (self.w - 20, self.h), (0, 180, 255), 4)

                if self.finish_line_obj is not None:
                    fy = int(self.finish_line_obj["y"])
                    if -100 <= fy <= self.h + 100:
                        cv2.rectangle(img, (40, fy - 12), (self.w - 40, fy + 12), (255, 255, 255), cv2.FILLED)
                        cv2.rectangle(img, (40, fy - 12), (self.w - 40, fy + 12), (0, 215, 255), 2)
                        cv2.putText(img, "REFLOW OVEN EXIT / FINISH LINE", (self.w // 2 - 200, fy + 6), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 24, 33), 2)

                for obs in self.obstacles:
                    if obs["img"] is not None:
                        img = overlay_transparent(img, obs["img"], obs["x"] - obs["r"], obs["y"] - obs["r"])
                    else:
                        cv2.circle(img, (obs["x"], int(obs["y"])), obs["r"], (0, 100, 255), cv2.FILLED)

                rx = int(self.robot_x)
                if self.hit_flash_timer > 0 and (self.hit_flash_timer // 5) % 2 == 0:
                    if self.robot_img is not None:
                        red_robot = self.robot_img.copy()
                        red_robot[:, :, 0] = 0 
                        red_robot[:, :, 1] = 0 
                        img = overlay_transparent(img, red_robot, rx - self.robot_radius, self.robot_y - self.robot_radius)
                    else:
                        cv2.circle(img, (rx, self.robot_y), self.robot_radius, (0, 0, 255), cv2.FILLED)
                else:
                    if self.robot_img is not None:
                        img = overlay_transparent(img, self.robot_img, rx - self.robot_radius, self.robot_y - self.robot_radius)
                    else:
                        cv2.circle(img, (rx, self.robot_y), self.robot_radius, (28, 41, 218), cv2.FILLED)

                # --- WEBCAM PICTURE-IN-PICTURE (PiP) FEED ---
                success_pip, pip_frame = self.stream.read()
                if success_pip and pip_frame is not None:
                    pip_frame = cv2.flip(pip_frame, 1)
                    pip_w, pip_h = 240, 135
                    pip_resized = cv2.resize(pip_frame, (pip_w, pip_h))
                    
                    if results and results.pose_landmarks:
                        lms = results.pose_landmarks[0]
                        lx_p = int(lms[11].x * pip_w)
                        ly_p = int(lms[11].y * pip_h)
                        rx_p = int(lms[12].x * pip_w)
                        ry_p = int(lms[12].y * pip_h)
                        cv2.circle(pip_resized, (lx_p, ly_p), 4, (0, 255, 128), -1)
                        cv2.circle(pip_resized, (rx_p, ry_p), 4, (0, 255, 128), -1)
                        cv2.line(pip_resized, (lx_p, ly_p), (rx_p, ry_p), (0, 255, 128), 2)

                    pip_x, pip_y = 30, self.h - pip_h - 35
                    draw_rounded_rect(img, (pip_x - 3, pip_y - 3), (pip_x + pip_w + 3, pip_y + pip_h + 3), (0, 180, 255), 1, 8)
                    img[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = pip_resized
                    cv2.putText(img, "LIVE TRACKING", (pip_x + 8, pip_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # --- MODERN GUI OVERLAYS ---
                if self.game_state == "START_SCREEN":
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (self.w, self.h), (10, 12, 18), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

                    draw_rounded_rect(img, (self.w//2 - 460, self.h//2 - 200), (self.w//2 + 460, self.h//2 + 180), (45, 55, 75), cv2.FILLED, 20)
                    draw_rounded_rect(img, (self.w//2 - 460, self.h//2 - 200), (self.w//2 + 460, self.h//2 + 180), (0, 180, 255), 2, 20)

                    cv2.putText(img, "PLEXUS FACTORY NAVIGATOR", (self.w//2 - 320, self.h//2 - 130), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 215, 255), 2)
                    cv2.putText(img, "GOAL: Guide your PCB safely through the reflow oven to 100m!", (self.w//2 - 390, self.h//2 - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 1)
                    cv2.putText(img, "CONTROLS: Lean Left or Right with your shoulders to steer.", (self.w//2 - 380, self.h//2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 200, 220), 1)

                    is_tracking = results is not None and results.pose_landmarks is not None and len(results.pose_landmarks) > 0
                    track_status = "PLAYER DETECTED - READY!" if is_tracking else "SEARCHING FOR PLAYER (STAND BACK)..."
                    track_color = (0, 255, 128) if is_tracking else (0, 165, 255)
                    cv2.putText(img, track_status, (self.w//2 - 210, self.h//2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 1)

                    cv2.putText(img, "Press [SPACEBAR] or [ENTER] to Start", (self.w//2 - 220, self.h//2 + 115), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)

                elif self.game_state == "PLAYING":
                    overlay_hud = img.copy()
                    draw_rounded_rect(overlay_hud, (30, 20), (self.w - 30, 100), (30, 38, 52), cv2.FILLED, 12)
                    cv2.addWeighted(overlay_hud, 0.8, img, 0.2, 0, img)
                    draw_rounded_rect(img, (30, 20), (self.w - 30, 100), (60, 80, 110), 1, 12)

                    cv2.putText(img, f"STATUS: {steering_command}", (50, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, steering_color, 2, cv2.LINE_AA)

                    gauge_x, gauge_y, gauge_w = 50, 75, 220
                    cv2.rectangle(img, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + 10), (20, 24, 33), cv2.FILLED)
                    cv2.rectangle(img, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + 10), (70, 90, 110), 1)
                    
                    center_tick = gauge_x + gauge_w // 2
                    max_range = 0.12
                    lean_thresh = 0.05
                    
                    thresh_offset_px = int((lean_thresh / max_range) * (gauge_w / 2))
                    left_thresh_x = center_tick - thresh_offset_px
                    right_thresh_x = center_tick + thresh_offset_px

                    cv2.line(img, (left_thresh_x, gauge_y - 2), (left_thresh_x, gauge_y + 12), (255, 200, 0), 2)
                    cv2.line(img, (right_thresh_x, gauge_y - 2), (right_thresh_x, gauge_y + 12), (255, 200, 0), 2)
                    
                    clamped_diff = max(-max_range, min(max_range, shoulder_lean_diff))
                    indicator_offset = int((clamped_diff / max_range) * (gauge_w / 2))
                    ind_x = center_tick + indicator_offset
                    cv2.circle(img, (ind_x, gauge_y + 5), 5, steering_color, cv2.FILLED)

                    bar_x1, bar_y1, bar_x2, bar_y2 = 420, 45, 860, 70
                    draw_rounded_rect(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (20, 24, 33), cv2.FILLED, 10)
                    filled_w = int((bar_x2 - bar_x1) * (self.distance / self.max_distance))
                    
                    if filled_w > 0:
                        draw_rounded_rect(img, (bar_x1, bar_y1), (bar_x1 + filled_w, bar_y2), (0, 180, 255), cv2.FILLED, 10)
                    draw_rounded_rect(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (90, 110, 140), 1, 10)
                    
                    cv2.putText(img, f"OVEN PROGRESS: {int(self.distance)}m / {int(self.max_distance)}m", (bar_x1 + 35, bar_y1 + 17), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.putText(img, "HP:", (self.w - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    for h_idx in range(3):
                        color = (0, 255, 128) if h_idx < self.health else (60, 70, 90)
                        hx = self.w - 230 + (h_idx * 45)
                        draw_rounded_rect(img, (hx, 45), (hx + 35, 75), color, cv2.FILLED, 6)

                elif self.game_state == "GAME_OVER":
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (self.w, self.h), (10, 10, 15), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

                    draw_rounded_rect(img, (self.w//2 - 360, self.h//2 - 160), (self.w//2 + 360, self.h//2 + 140), (40, 25, 30), cv2.FILLED, 16)
                    draw_rounded_rect(img, (self.w//2 - 360, self.h//2 - 160), (self.w//2 + 360, self.h//2 + 140), (0, 0, 255), 2, 16)

                    cv2.putText(img, "SYSTEM FAILURE", (self.w//2 - 210, self.h//2 - 80), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
                    cv2.putText(img, f"FINAL DISTANCE: {int(self.distance)}m", (self.w//2 - 180, self.h//2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                    cv2.putText(img, "Press 'R' to Retry  |  Press 'ESC' to Quit", (self.w//2 - 230, self.h//2 + 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 170, 170), 1)

                elif self.game_state == "WIN":
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (self.w, self.h), (10, 15, 20), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

                    draw_rounded_rect(img, (self.w//2 - 380, self.h//2 - 160), (self.w//2 + 380, self.h//2 + 140), (25, 45, 35), cv2.FILLED, 16)
                    draw_rounded_rect(img, (self.w//2 - 380, self.h//2 - 160), (self.w//2 + 380, self.h//2 + 140), (0, 255, 128), 2, 16)

                    cv2.putText(img, "REFLOW COMPLETE!", (self.w//2 - 240, self.h//2 - 80), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 128), 2)
                    cv2.putText(img, "Successfully reached 100 meters!", (self.w//2 - 230, self.h//2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(img, "Press 'R' to Play Again  |  Press 'ESC' to Quit", (self.w//2 - 250, self.h//2 + 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 170, 170), 1)

                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time
                instant_fps = 1 / dt if dt > 0 else 0
                smoothed_fps = lerp(smoothed_fps, instant_fps, 0.1)

                

                if self.debug_mode:
                    debug_overlay = img.copy()
                    draw_rounded_rect(debug_overlay, (30, 120), (450, 320), (15, 20, 30), cv2.FILLED, 10)
                    cv2.addWeighted(debug_overlay, 0.75, img, 0.25, 0, img)
                    draw_rounded_rect(img, (30, 120), (450, 320), (0, 180, 255), 1, 10)

                    cv2.putText(img, "--- TELEMETRY DEBUG HUD ---", (45, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 1)
                    cv2.putText(img, f"Render FPS: {int(smoothed_fps)}", (45, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)
                    cv2.putText(img, f"Frame Delta (dt): {dt*1000:.1f} ms", (45, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                    cv2.putText(img, f"Active Obstacles: {len(self.obstacles)}", (45, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                    cv2.putText(img, f"Belt Speed: {self.belt_speed:.2f}", (45, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                    
                    left_s_val = f"{results.pose_landmarks[0][11].y:.3f}" if (results and results.pose_landmarks) else "N/A"
                    right_s_val = f"{results.pose_landmarks[0][12].y:.3f}" if (results and results.pose_landmarks) else "N/A"
                    cv2.putText(img, f"Shoulder Y (L/R): {left_s_val} / {right_s_val}", (45, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                    cv2.putText(img, f"Inference Thread Active: {self.inference_thread.is_alive()}", (45, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128) if self.inference_thread.is_alive() else (0, 0, 255), 1)

                target_h = int(self.w * (10 / 16))
                canvas = np.zeros((target_h, self.w, 3), dtype=np.uint8)
                y_offset = (target_h - self.h) // 2
                canvas[y_offset:y_offset+self.h, 0:self.w] = img

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
