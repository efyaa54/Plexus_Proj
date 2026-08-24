import cv2
import numpy as np
import mediapipe as mp
import time
import threading
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
            except Exception:
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
class AOIInspectorGame:
    def __init__(self):
        self.w, self.h = 1280, 720
        self.window_name = "Plexus AOI Inspector"

        self.player_data = load_player_data()
        self.score_awarded = False

        # --- GAME STATE ---
        self.game_state = "START_SCREEN"
        self.score = 0
        self.current_board_idx = 0
        
        # Setup: 3 Good (False), 2 Bad (True with random bad image index 0 or 1)
        self.boards = [
            {"faulty": False, "bad_idx": 0},
            {"faulty": False, "bad_idx": 0},
            {"faulty": False, "bad_idx": 0},
            {"faulty": True, "bad_idx": 0},
            {"faulty": True, "bad_idx": 1},
        ]
        random.shuffle(self.boards)
        
        self.feedback_msg = ""
        self.feedback_color = (0, 0, 0)
        self.feedback_timer = 0
        self.debug_mode = False

        # --- GESTURE STATE (FSM) ---
        self.current_gesture = "None"
        self.gesture_hold_frames = 0
        self.CONFIRM_FRAMES = 35 

        # --- ASSET LOADING ---
        self.load_graphics()

        # --- THREADING & ML ---
        self.setup_window()
        self.stream = WebcamStream(src=0, width=self.w, height=self.h)

        self.latest_result = None
        self.inference_lock = threading.Lock()
        self.inference_running = True

        base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task', delegate=python.BaseOptions.Delegate.CPU)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()

    def generate_synthetic_board(self, is_faulty, w, h):
        """Fallback generator scaled dynamically to requested dimensions."""
        board = np.full((h, w, 3), (30, 80, 20), dtype=np.uint8)
        cv2.rectangle(board, (10, 10), (w-10, h-10), (100, 150, 100), 4) 
        
        positions = [(int(w*0.25), int(h*0.25)), (int(w*0.75), int(h*0.25)), 
                     (int(w*0.25), int(h*0.75)), (int(w*0.75), int(h*0.75))]
        faulty_index = random.randint(0, 3) if is_faulty else -1
        
        for i, (x, y) in enumerate(positions):
            if i == faulty_index:
                pts = np.array([[x-30, y-60], [x+80, y-30], [x+60, y+80], [x-60, y+60]], np.int32)
                cv2.fillPoly(board, [pts], (0, 0, 200)) 
                cv2.polylines(board, [pts], True, (0, 0, 255), 4)
                cv2.putText(board, "ERR", (x-20, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            else:
                cv2.rectangle(board, (x-60, y-60), (x+60, y+60), (50, 50, 50), -1)
                cv2.rectangle(board, (x-60, y-60), (x+60, y+60), (150, 150, 150), 3)
        return board

    def load_graphics(self):
        self.board_good = cv2.imread("board_good.png")
        
        # Load multiple bad board variations
        self.board_bads = []
        for filename in ["board_bad.png", "board_bad2.png"]:
            img = cv2.imread(filename)
            if img is not None:
                self.board_bads.append(img)
        
        # Determine the maximum footprint available for the board on the left side
        max_w, max_h = 750, 560
        
        # Dynamically scale retaining exact aspect ratio
        if self.board_good is not None:
            orig_h, orig_w = self.board_good.shape[:2]
            scale = min(max_w / orig_w, max_h / orig_h)
            self.board_good = cv2.resize(self.board_good, (int(orig_w * scale), int(orig_h * scale)))
            
        # Dynamically scale all bad boards
        scaled_bads = []
        for bad_img in self.board_bads:
            orig_h, orig_w = bad_img.shape[:2]
            scale = min(max_w / orig_w, max_h / orig_h)
            scaled_bads.append(cv2.resize(bad_img, (int(orig_w * scale), int(orig_h * scale))))
        self.board_bads = scaled_bads

        self.synth_good = self.generate_synthetic_board(False, max_w, max_h)
        self.synth_bad = self.generate_synthetic_board(True, max_w, max_h)

    def get_current_board_img(self):
        board_data = self.boards[self.current_board_idx]
        if board_data["faulty"]:
            if self.board_bads:
                # Pick the specific bad board variant for this round
                idx = board_data["bad_idx"] % len(self.board_bads)
                return self.board_bads[idx]
            else:
                return self.synth_bad
        else:
            return self.board_good if self.board_good is not None else self.synth_good

    def setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    def reset_game(self):
        self.score = 0
        self.current_board_idx = 0
        self.boards = [
            {"faulty": False, "bad_idx": 0},
            {"faulty": False, "bad_idx": 0},
            {"faulty": False, "bad_idx": 0},
            {"faulty": True, "bad_idx": 0},
            {"faulty": True, "bad_idx": 1},
        ]
        random.shuffle(self.boards)
        self.game_state = "PLAYING"
        self.gesture_hold_frames = 0
        self.current_gesture = "None"
        self.score_awarded = False

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
                result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
                with self.inference_lock:
                    self.latest_result = result
            except Exception:
                pass

            time.sleep(0.001)

    def reset_game(self):
        self.score = 0
        self.current_board_idx = 0
        random.shuffle(self.boards)
        self.game_state = "PLAYING"
        self.gesture_hold_frames = 0
        self.current_gesture = "None"
        self.score_awarded = False

    def run(self):
        prev_time = time.time()
        smoothed_fps = 0.0

        try:
            while True:
                with self.inference_lock:
                    results = self.latest_result

                img = np.full((self.h, self.w, 3), (20, 24, 33), dtype=np.uint8)
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27: 
                    break
                elif key == ord('d') or key == ord('D'):
                    self.debug_mode = not self.debug_mode

                recognized_gesture = "None"
                if results and results.gestures:
                    recognized_gesture = results.gestures[0][0].category_name

                # --- START SCREEN ---
                if self.game_state == "START_SCREEN":
                    if key == ord(' ') or key == 13: 
                        self.reset_game()
                        
                    draw_rounded_rect(img, (self.w//2 - 460, self.h//2 - 200), (self.w//2 + 460, self.h//2 + 180), (45, 55, 75), cv2.FILLED, 20)
                    draw_rounded_rect(img, (self.w//2 - 460, self.h//2 - 200), (self.w//2 + 460, self.h//2 + 180), (0, 255, 128), 2, 20)
                    cv2.putText(img, "PLEXUS AOI INSPECTOR", (self.w//2 - 280, self.h//2 - 130), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 128), 2)
                    cv2.putText(img, "GOAL: Inspect 5 PCBs. Approve the good, reject the defective.", (self.w//2 - 390, self.h//2 - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 1)
                    cv2.putText(img, "CONTROLS: Show 'Thumbs Up' to Pass, 'Thumbs Down' to Reject.", (self.w//2 - 410, self.h//2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 200, 220), 1)
                    cv2.putText(img, "Press [SPACEBAR] or [ENTER] to Start", (self.w//2 - 220, self.h//2 + 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)

                # --- PLAYING STATE ---
                elif self.game_state == "PLAYING":
                    # 1. Top Status Bar
                    draw_rounded_rect(img, (40, 20), (self.w - 40, 100), (30, 38, 52), cv2.FILLED, 12)
                    draw_rounded_rect(img, (40, 20), (self.w - 40, 100), (60, 80, 110), 1, 12)
                    cv2.putText(img, f"INSPECTING BOARD {self.current_board_idx + 1}/5", (70, 65), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(img, f"SCORE: {self.score}", (self.w - 220, 65), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 128), 2)

                    active = self.player_data["active_player"]
                    player_text = f"PLAYER: {active['name']}  |  SCORE: {active['score']}"
                    cv2.putText(img, player_text, (self.w - 600, 65), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 128), 1)

                    # 2. Render Scaled Board Image
                    board_img = self.get_current_board_img()
                    bh, bw = board_img.shape[:2]
                    
                    bx = 50 
                    by = (self.h - bh) // 2 + 40 # Shift down slightly to clear the header
                    
                    img[by:by+bh, bx:bx+bw] = board_img
                    draw_rounded_rect(img, (bx-4, by-4), (bx+bw+4, by+bh+4), (100, 120, 150), 3, 10)

                    # 3. GUI Operator Dashboard (Placed securely to the right of the board)
                    panel_x = bx + bw + 40
                    panel_w = (self.w - 40) - panel_x
                    
                    draw_rounded_rect(img, (panel_x, by), (panel_x + panel_w, by + bh), (30, 38, 52), cv2.FILLED, 15)
                    draw_rounded_rect(img, (panel_x, by), (panel_x + panel_w, by + bh), (60, 80, 110), 2, 15)
                    
                    text_x = panel_x + 30
                    text_y = by + 50
                    cv2.putText(img, "OPERATOR DECISION:", (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                    # Color-coded action cards
                    card_w = panel_w - 60
                    tu_y = text_y + 30
                    td_y = tu_y + 90
                    
                    draw_rounded_rect(img, (text_x, tu_y), (text_x + card_w, tu_y + 70), (20, 45, 30), cv2.FILLED, 8)
                    draw_rounded_rect(img, (text_x, tu_y), (text_x + card_w, tu_y + 70), (0, 255, 128), 2, 8)
                    cv2.putText(img, "PASS", (text_x + 20, tu_y + 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 128), 2)
                    cv2.putText(img, "(Thumbs Up)", (text_x + 130, tu_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 1)

                    draw_rounded_rect(img, (text_x, td_y), (text_x + card_w, td_y + 70), (45, 20, 20), cv2.FILLED, 8)
                    draw_rounded_rect(img, (text_x, td_y), (text_x + card_w, td_y + 70), (0, 0, 255), 2, 8)
                    cv2.putText(img, "REJECT", (text_x + 20, td_y + 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(img, "(Thumbs Down)", (text_x + 160, td_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 1)

                    # 4. Debounce Math & Loading Bar
                    if recognized_gesture in ["Thumb_Up", "Thumb_Down"]:
                        if recognized_gesture == self.current_gesture:
                            self.gesture_hold_frames += 1
                        else:
                            self.current_gesture = recognized_gesture
                            self.gesture_hold_frames = 1
                    else:
                        self.gesture_hold_frames = 0
                        self.current_gesture = "None"

                    bar_y = td_y + 100
                    draw_rounded_rect(img, (text_x, bar_y), (text_x + card_w, bar_y + 30), (15, 20, 30), cv2.FILLED, 8)
                    
                    if self.gesture_hold_frames > 0:
                        fill_w = int((self.gesture_hold_frames / self.CONFIRM_FRAMES) * card_w)
                        fill_w = min(fill_w, card_w)
                        
                        if fill_w > 10:
                            color = (0, 255, 128) if self.current_gesture == "Thumb_Up" else (0, 0, 255)
                            draw_rounded_rect(img, (text_x, bar_y), (text_x + fill_w, bar_y + 30), color, cv2.FILLED, 8)
                            
                        cv2.putText(img, f"Locking {self.current_gesture}...", (text_x, bar_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # 5. Picture-in-Picture (Embedded inside the operator dashboard!)
                    success_pip, pip_frame = self.stream.read()
                    if success_pip and pip_frame is not None:
                        pip_frame = cv2.flip(pip_frame, 1)
                        # Scale PiP to perfectly fit the bottom of the card
                        pip_w = card_w
                        pip_h = int(pip_w * (9 / 16))
                        pip_resized = cv2.resize(pip_frame, (pip_w, pip_h))
                        
                        if results and results.hand_landmarks:
                            lms = results.hand_landmarks[0]
                            for start_idx, end_idx in HAND_CONNECTIONS:
                                x1, y1 = int(lms[start_idx].x * pip_w), int(lms[start_idx].y * pip_h)
                                x2, y2 = int(lms[end_idx].x * pip_w), int(lms[end_idx].y * pip_h)
                                cv2.line(pip_resized, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            for lm in lms:
                                cx, cy = int(lm.x * pip_w), int(lm.y * pip_h)
                                cv2.circle(pip_resized, (cx, cy), 4, (0, 215, 255), -1)

                        pip_y = (by + bh) - pip_h - 20
                        draw_rounded_rect(img, (text_x - 3, pip_y - 3), (text_x + pip_w + 3, pip_y + pip_h + 3), (100, 120, 150), 2, 8)
                        img[pip_y:pip_y+pip_h, text_x:text_x+pip_w] = pip_resized
                        cv2.putText(img, "LIVE CAMERA FEED", (text_x, pip_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # 6. Final Decision Logic
                    if self.gesture_hold_frames >= self.CONFIRM_FRAMES:
                        is_faulty = self.boards[self.current_board_idx]["faulty"]
                        if (self.current_gesture == "Thumb_Up" and not is_faulty) or \
                           (self.current_gesture == "Thumb_Down" and is_faulty):
                            self.score += 1
                            self.feedback_msg = "CORRECT DECISION!"
                            self.feedback_color = (0, 255, 128)
                        else:
                            self.feedback_msg = "WRONG! AOI FAILED!"
                            self.feedback_color = (0, 0, 255)
                        
                        self.game_state = "FEEDBACK"
                        self.feedback_timer = 90
                        self.gesture_hold_frames = 0

                # --- FEEDBACK STATE ---
                elif self.game_state == "FEEDBACK":
                    board_img = self.get_current_board_img()
                    bh, bw = board_img.shape[:2]
                    bx = 50 
                    by = (self.h - bh) // 2 + 40
                    img[by:by+bh, bx:bx+bw] = board_img
                    draw_rounded_rect(img, (bx-4, by-4), (bx+bw+4, by+bh+4), (100, 120, 150), 3, 10)
                    
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
                    
                    # Draw a nice backing plate for the feedback text
                    draw_rounded_rect(img, (self.w//2 - 320, self.h//2 - 60), (self.w//2 + 320, self.h//2 + 60), (20, 24, 33), cv2.FILLED, 15)
                    draw_rounded_rect(img, (self.w//2 - 320, self.h//2 - 60), (self.w//2 + 320, self.h//2 + 60), self.feedback_color, 3, 15)
                    
                    # Center the text
                    text_size = cv2.getTextSize(self.feedback_msg, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4)[0]
                    text_x = (self.w - text_size[0]) // 2
                    cv2.putText(img, self.feedback_msg, (text_x, self.h//2 + 15), cv2.FONT_HERSHEY_DUPLEX, 1.5, self.feedback_color, 4)
                    
                    self.feedback_timer -= 1
                    if self.feedback_timer <= 0:
                        self.current_board_idx += 1
                        if self.current_board_idx >= len(self.boards):
                            self.game_state = "GAME_OVER"
                        else:
                            self.game_state = "PLAYING"

                # --- GAME OVER STATE ---
                elif self.game_state == "GAME_OVER":

                    if not self.score_awarded:
                        # Award points based on correct inspections
                        if self.score == len(self.boards): 
                            self.player_data["active_player"]["score"] += 1
                            save_player_data(self.player_data)
                            self.score_awarded = True

                        
                    draw_rounded_rect(img, (self.w//2 - 360, self.h//2 - 160), (self.w//2 + 360, self.h//2 + 140), (25, 35, 45), cv2.FILLED, 16)
                    draw_rounded_rect(img, (self.w//2 - 360, self.h//2 - 160), (self.w//2 + 360, self.h//2 + 140), (0, 215, 255), 2, 16)

                    cv2.putText(img, "BOARD INSPECTION COMPLETE", (self.w//2 - 310, self.h//2 - 80), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
                    cv2.putText(img, f"FINAL SCORE: {self.score} / {len(self.boards)}", (self.w//2 - 180, self.h//2 + 10), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 215, 255), 2)
                    cv2.putText(img, "Press 'R' to Retry  |  Press 'ESC' to Quit", (self.w//2 - 240, self.h//2 + 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 170, 170), 1)
                                
                    if key == ord('r') or key == ord('R'):
                        self.reset_game()

                # --- FPS & CANVAS FORMATTING ---
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time
                instant_fps = 1 / dt if dt > 0 else 0
                smoothed_fps = lerp(smoothed_fps, instant_fps, 0.1)

                if self.debug_mode:
                    cv2.putText(img, f"FPS: {int(smoothed_fps)} | Gesture: {recognized_gesture}", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1)

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
    game = AOIInspectorGame()
    game.run()
