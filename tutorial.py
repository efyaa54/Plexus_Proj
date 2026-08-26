"""
Plexus AI Arcade - Interactive How to Play (Authentic Mechanics Edition)
A self-contained Flet walkthrough that exactly mirrors the game logic loops,
teaching the player the precise computer vision interactions required.
"""

from __future__ import annotations

import os
import time
import base64
import asyncio
import math
from dataclasses import dataclass

import cv2
import numpy as np
import flet as ft
import psutil
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)  # OpenCV uses BGR

# --------------------------------------------------------------------------
# Shared visual language
# --------------------------------------------------------------------------

APP_TITLE = "Plexus AI Arcade - How to Play"
SHUTDOWN_GRACE_S = 0.2
CHILD_TERMINATE_TIMEOUT_S = 2.0
FRAME_INTERVAL_S = 0.03  # ~33fps target cadence between processed frames

PLEXUS_RED = "#DA291C"
PLEXUS_CHARCOAL = "#1A1A1A"
PLEXUS_GRAY = "#333333"
PLEXUS_WHITE = "#FFFFFF"
PLEXUS_GREEN = "#2ECC71"


def kill_process_tree() -> None:
    try:
        children = psutil.Process(os.getpid()).children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=CHILD_TERMINATE_TIMEOUT_S)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def lerp(current, target, alpha=0.3):
    return current + alpha * (target - current)


# --------------------------------------------------------------------------
# Content Pages 
# --------------------------------------------------------------------------

@dataclass
class TutorialPage:
    game_title: str
    subtitle: str
    objective: str
    controls: list[str]
    shortcuts: list[str]
    accent: str = PLEXUS_RED


PAGES: list[TutorialPage] = [
    TutorialPage(
        game_title="System Calibration",
        subtitle="Operator Setup",
        objective=(
            "Every station in the Plexus AI Arcade tracks your body in 3D space. "
            "Ensure the system can see you clearly before starting."
        ),
        controls=[
            "Stand about arm's length from the camera",
            "Keep your hand inside the frame with fingers spread",
            "Verify both the hand and upper-body skeletons map cleanly to your joints",
        ],
        shortcuts=["Esc closes the current game and returns to the arcade hub"],
        accent=PLEXUS_GRAY,
    ),
    TutorialPage(
        game_title="SMT Pick and Place",
        subtitle="Circuit Builder",
        objective="Learn the dynamic pinch scaling calculation used to manipulate components.",
        controls=[
            "Keep your palm open facing the camera",
            "Pinch using ONLY your thumb and index finger",
            "Maintain your other three fingers OPEN so the camera tracks the pinch accurately",
            "Grab and place all 3 components into their respective sockets sequentially",
        ],
        shortcuts=["R resets the sequence if a component gets stuck"],
        accent=PLEXUS_RED,
    ),
    TutorialPage(
        game_title="SMT Reflow",
        subtitle="Factory Navigator",
        objective="Learn the center-of-gravity tracking used to steer the PCB down the conveyor.",
        controls=[
            "Stand back so your shoulders AND hips are visible to the camera",
            "The system calculates lean by comparing your shoulders to your hips (the green spine)",
            "Keep your hips planted and lean your torso LEFT to hit the first target",
            "Then, lean RIGHT to hit the second target",
        ],
        shortcuts=["Space starts the run in the actual game"],
        accent="#F39C12",
    ),
    TutorialPage(
        game_title="AOI Inspection",
        subtitle="AOI Inspector",
        objective="Learn the frame-debouncing logic used to prevent accidental decisions.",
        controls=[
            "Make a clear 'Thumbs Up' gesture to approve the board",
            "Hold the gesture perfectly still to fill the confidence ring",
            "Next, make a 'Thumbs Down' gesture to reject the board",
        ],
        shortcuts=["The system requires 35 consecutive frames to lock a decision"],
        accent=PLEXUS_GREEN,
    ),
    TutorialPage(
        game_title="X-Ray Inspection",
        subtitle="Defect Detective",
        objective="Combine steering and pinching to operate the X-ray lens.",
        controls=[
            "Move your open hand to steer the X-ray lens across the board",
            "When you spot the red hidden defect, hover the lens over it",
            "Perform a sharp pinch gesture to successfully flag the defect",
        ],
        shortcuts=["Find all hidden defects to pass the actual game"],
        accent="#3498DB",
    ),
]

# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

class TutorialApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.index = 0
        self._shutting_down = False

        self.video_image = ft.Image(
            src=None, fit=ft.BoxFit.CONTAIN, expand=True, gapless_playback=True,
        )

        # Authentic Game State Variables
        self.cursor_x, self.cursor_y = 320.0, 240.0
        
        # Game 1: Circuit Builder (Multi-Stage)
        self.cb_stage = 0
        self.cb_stages = [
            (150, 150, 480, 150, (255, 100, 50)),   # Top: L to R (Blue-ish)
            (150, 350, 480, 350, (200, 50, 200)),   # Bottom: L to R (Purple)
            (480, 240, 150, 240, (50, 200, 255))    # Middle: R to L (Yellow-ish)
        ]
        self.cb_comp_x, self.cb_comp_y = self.cb_stages[0][:2]
        self.cb_is_grabbed = False
        self.cb_success = False

        # Game 2: Factory Navigator (Left then Right)
        self.fn_lean_offset = 0.0
        self.fn_stage = 0 # 0 = Left, 1 = Right
        self.fn_success = False

        # Game 3: AOI Inspector (Up then Down)
        self.aoi_frames = 0
        self.AOI_CONFIRM_FRAMES = 35
        self.aoi_stage = 0 # 0 = Thumbs Up, 1 = Thumbs Down
        self.aoi_success = False

        # Game 4: Defect Detective
        self.dd_defect_x, self.dd_defect_y = 450, 300
        self.dd_success = False

        self._init_mediapipe()
        self._build_ui()

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._video_start_time = time.time()
        self._last_ts = -1
        self.page.run_task(self._video_loop)

    def _init_mediapipe(self):
        base_opt_h = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base_opt_h, num_hands=1, running_mode=vision.RunningMode.VIDEO)
        )
        base_opt_g = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        self.gest_detector = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(base_options=base_opt_g, num_hands=1, running_mode=vision.RunningMode.VIDEO)
        )
        base_opt_p = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base_opt_p, output_segmentation_masks=False, running_mode=vision.RunningMode.VIDEO)
        )

    def _build_ui(self):
        self._configure_window()
        self.left_column = ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=30,
        )
        self.left_wrapper = ft.Container(expand=4, content=self.left_column, alignment=ft.Alignment.CENTER)

        self.video_container = ft.Container(
            expand=6, border_radius=24, border=ft.Border.all(4, PLEXUS_RED),
            clip_behavior=ft.ClipBehavior.HARD_EDGE, content=self.video_image, alignment=ft.Alignment.CENTER,
        )

        self.body = ft.Row(
            expand=True, spacing=30, alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[self.left_wrapper, self.video_container],
        )

        self.page.add(ft.Container(expand=True, padding=50, alignment=ft.Alignment(0, 0), content=self.body))
        self._render()

    def _reset_games(self):
        """Resets all minigame state logic"""
        self.cb_stage = 0
        self.cb_comp_x, self.cb_comp_y = self.cb_stages[0][:2]
        self.cb_is_grabbed = False
        self.cb_success = False

        self.fn_stage = 0
        self.fn_success = False

        self.aoi_stage = 0
        self.aoi_frames = 0
        self.aoi_success = False

        self.dd_success = False

    def _capture_and_process(self) -> str | None:
        ret, frame = self.cap.read()
        if not ret: return None

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        is_pinching = False
        pinch_ratio = 1.0

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.time() - self._video_start_time) * 1000)
            if ts <= self._last_ts: ts = self._last_ts + 1
            self._last_ts = ts

            hand_res = self.hand_detector.detect_for_video(mp_img, ts)
            gest_res = self.gest_detector.recognize_for_video(mp_img, ts)
            pose_res = self.pose_detector.detect_for_video(mp_img, ts)
            
            idx = self.index

            # --- 1. BODY/POSE TRACKING (True Lean Mechanics) ---
            if pose_res and pose_res.pose_landmarks:
                pose_lms = pose_res.pose_landmarks[0]
                
                # Shoulders (11, 12) and Hips (23, 24) for true center of gravity lean
                mid_shoulder_x = (pose_lms[11].x + pose_lms[12].x) / 2.0
                mid_shoulder_y = (pose_lms[11].y + pose_lms[12].y) / 2.0
                
                mid_hip_x = (pose_lms[23].x + pose_lms[24].x) / 2.0
                mid_hip_y = (pose_lms[23].y + pose_lms[24].y) / 2.0
                
                # Lean is the offset between shoulders and hips, multiplied for sensitivity
                raw_lean = (mid_shoulder_x - mid_hip_x) * 4.0 
                target_lean = max(-1.2, min(1.2, raw_lean)) # Clamp to prevent escaping frame
                
                self.fn_lean_offset = lerp(self.fn_lean_offset, target_lean, 0.2)

                if idx in [0, 2]:
                    # Draw upper body bounding box/frame
                    for s, e in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]:
                        cx1, cy1 = int(pose_lms[s].x * w), int(pose_lms[s].y * h)
                        cx2, cy2 = int(pose_lms[e].x * w), int(pose_lms[e].y * h)
                        cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 150, 0), 2)
                        cv2.circle(frame, (cx1, cy1), 4, (255, 200, 100), -1)
                        cv2.circle(frame, (cx2, cy2), 4, (255, 200, 100), -1)
                    
                    # Draw True Center of Gravity (Neon Green Spine Vector)
                    sx, sy = int(mid_shoulder_x * w), int(mid_shoulder_y * h)
                    hx, hy = int(mid_hip_x * w), int(mid_hip_y * h)
                    cv2.line(frame, (hx, hy), (sx, sy), (0, 255, 128), 5)
                    cv2.circle(frame, (sx, sy), 8, (0, 255, 128), -1)

            # --- 2. HAND TRACKING ---
            if hand_res and hand_res.hand_landmarks:
                lms = hand_res.hand_landmarks[0]
                
                target_x = max(0, min(w, lms[9].x * w))
                target_y = max(0, min(h, lms[9].y * h))
                self.cursor_x = lerp(self.cursor_x, target_x, 0.4)
                self.cursor_y = lerp(self.cursor_y, target_y, 0.4)

                hand_scale = max(math.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y), 1e-4)
                pinch_ratio = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y) / hand_scale
                is_pinching = pinch_ratio < 0.35 

                if idx in [0, 1, 3, 4]:
                    for s, e in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                                 (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
                                 (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]:
                        cx1, cy1 = int(lms[s].x * w), int(lms[s].y * h)
                        cx2, cy2 = int(lms[e].x * w), int(lms[e].y * h)
                        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 128), 2)
                        cv2.circle(frame, (cx1, cy1), 4, (255, 255, 255), -1)

            # --- RENDER TUTORIAL STATES ---

            # STATE 0: Welcome Calibration
            if idx == 0:
                cv2.putText(frame, "STATUS: TRACKING ACTIVE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)

            # STATE 1: Circuit Builder Logic (Multi-stage)
            elif idx == 1:
                if self.cb_success:
                    tx, ty, tcolor = self.cb_stages[-1][2], self.cb_stages[-1][3], self.cb_stages[-1][4]
                else:
                    tx, ty, tcolor = self.cb_stages[self.cb_stage][2], self.cb_stages[self.cb_stage][3], self.cb_stages[self.cb_stage][4]

                cv2.circle(frame, (tx, ty), 50, (0, 255, 128), 3)
                cv2.putText(frame, f"SOCKET {self.cb_stage + 1}/3", (tx - 40, ty - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)

                if not self.cb_is_grabbed and is_pinching and not self.cb_success:
                    if math.hypot(self.cursor_x - self.cb_comp_x, self.cursor_y - self.cb_comp_y) < 50:
                        self.cb_is_grabbed = True
                elif self.cb_is_grabbed and pinch_ratio > 0.55: # Authentic release threshold
                    self.cb_is_grabbed = False
                    if math.hypot(self.cb_comp_x - tx, self.cb_comp_y - ty) < 40:
                        if self.cb_stage < len(self.cb_stages) - 1:
                            self.cb_stage += 1
                            self.cb_comp_x, self.cb_comp_y = self.cb_stages[self.cb_stage][:2]
                        else:
                            self.cb_success = True
                            self.cb_comp_x, self.cb_comp_y = tx, ty # Snap to final socket

                if self.cb_is_grabbed:
                    self.cb_comp_x, self.cb_comp_y = int(self.cursor_x), int(self.cursor_y)

                c_color = (0, 255, 0) if self.cb_success else ((0, 255, 255) if self.cb_is_grabbed else tcolor)
                cv2.rectangle(frame, (self.cb_comp_x - 30, self.cb_comp_y - 30), (self.cb_comp_x + 30, self.cb_comp_y + 30), c_color, -1)
                
                if is_pinching:
                    cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 12, (0, 255, 0), cv2.FILLED)
                else:
                    cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 15, (255, 255, 255), 2)
                    
                if self.cb_success:
                    cv2.putText(frame, "ALL COMPONENTS PLACED!", (w//2 - 140, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

            # STATE 2: Factory Navigator Logic (Left -> Right)
            elif idx == 2:
                # The visual box's position is dictated by the true lean offset
                pcb_x = int(w/2 + self.fn_lean_offset * (w/2 - 80))
                
                # Conveyor lines
                cv2.line(frame, (w//2 - 120, 0), (w//2 - 250, h), (100, 100, 100), 2)
                cv2.line(frame, (w//2, 0), (w//2, h), (70, 70, 70), 2)
                cv2.line(frame, (w//2 + 120, 0), (w//2 + 250, h), (100, 100, 100), 2)
                
                # Target lane based on stage
                target_lane_x = w//2 + (-150 if self.fn_stage == 0 else 150)
                
                if not self.fn_success:
                    cv2.rectangle(frame, (target_lane_x - 60, h//2 + 60), (target_lane_x + 60, h//2 + 180), (0, 255, 128), 2)
                    cv2.putText(frame, "STEER HERE", (target_lane_x - 45, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
                
                # Collision Check
                if abs(pcb_x - target_lane_x) < 50:
                    if self.fn_stage == 0:
                        self.fn_stage = 1 # Move to Right stage
                    else:
                        self.fn_success = True

                # Draw Steering PCB
                pcb_color = (0, 255, 0) if self.fn_success else (0, 150, 255)
                cv2.rectangle(frame, (pcb_x - 45, h//2 + 80), (pcb_x + 45, h//2 + 160), pcb_color, -1)
                cv2.rectangle(frame, (pcb_x - 45, h//2 + 80), (pcb_x + 45, h//2 + 160), (255, 255, 255), 2)
                
                if self.fn_success:
                    cv2.putText(frame, "LEAN CALIBRATION SUCCESS!", (w//2 - 160, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                elif self.fn_stage == 0:
                    cv2.putText(frame, "<- LEAN LEFT", (w//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "LEAN RIGHT ->", (w//2 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # STATE 3: AOI Inspector Logic (Up -> Down)
            elif idx == 3:
                gest = "None"
                if gest_res and gest_res.gestures:
                    gest = gest_res.gestures[0][0].category_name

                target_gest = "Thumb_Up" if self.aoi_stage == 0 else "Thumb_Down"

                if gest == target_gest:
                    self.aoi_frames = min(self.AOI_CONFIRM_FRAMES, self.aoi_frames + 1)
                else:
                    self.aoi_frames = max(0, self.aoi_frames - 1)

                cv2.circle(frame, (w // 2, h // 2), 60, (50, 50, 50), 6)
                ang = int((self.aoi_frames / self.AOI_CONFIRM_FRAMES) * 360)
                if ang > 0:
                    cv2.ellipse(frame, (w // 2, h // 2), (60, 60), 0, -90, -90 + ang, (0, 255, 128), 6)
                
                if self.aoi_frames >= self.AOI_CONFIRM_FRAMES:
                    if self.aoi_stage == 0:
                        self.aoi_stage = 1
                        self.aoi_frames = 0
                    else:
                        self.aoi_success = True

                if self.aoi_success:
                    cv2.putText(frame, "ALL DECISIONS LOCKED!", (w // 2 - 110, h // 2 + 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 128), 2)
                else:
                    instruction = f"HOLD THUMBS UP ({int((self.aoi_frames/35)*100)}%)" if self.aoi_stage == 0 else f"HOLD THUMBS DOWN ({int((self.aoi_frames/35)*100)}%)"
                    cv2.putText(frame, instruction, (w // 2 - 100, h // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # STATE 4: Defect Detective Logic
            elif idx == 4:
                mask = np.zeros_like(frame)
                lens_radius = 80
                cv2.circle(mask, (int(self.cursor_x), int(self.cursor_y)), lens_radius, (0, 0, 150), -1)
                frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

                cv2.circle(frame, (self.dd_defect_x, self.dd_defect_y), 15, (0, 0, 255), -1)
                
                if is_pinching and math.hypot(self.cursor_x - self.dd_defect_x, self.cursor_y - self.dd_defect_y) < lens_radius:
                    self.dd_success = True

                if self.dd_success:
                    cv2.rectangle(frame, (self.dd_defect_x - 30, self.dd_defect_y - 30), (self.dd_defect_x + 30, self.dd_defect_y + 30), (0, 255, 0), 3)
                    cv2.putText(frame, "DEFECT FLAGGED!", (self.dd_defect_x - 70, self.dd_defect_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                l_color = (0, 255, 0) if is_pinching else (0, 215, 255)
                cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), lens_radius, l_color, 4)
                cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 5, l_color, -1)

        except Exception as e:
            print(f"Vision error: {e}")

        idx = self.index
        accent_bgr = _hex_to_bgr(PAGES[idx].accent)
        t = 4
        cv2.rectangle(frame, (t, t), (w - 1 - t, h - 1 - t), accent_bgr, 8)

        try:
            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok: return None
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception as e:
            return None

    async def _video_loop(self):
        while not self._shutting_down:
            loop_start = time.time()
            frame_data_uri = await asyncio.to_thread(self._capture_and_process)

            if frame_data_uri is not None and not self._shutting_down:
                self.video_image.src = frame_data_uri
                try: self.video_image.update()
                except Exception: pass

            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0.0, FRAME_INTERVAL_S - elapsed))

    def _configure_window(self) -> None:
        page = self.page
        page.title = APP_TITLE
        page.padding = 0
        page.bgcolor = PLEXUS_CHARCOAL
        page.window.full_screen = True
        page.window.resizable = False
        page.window.prevent_close = True
        page.window.on_event = self._on_window_event
        page.on_keyboard_event = self._on_keyboard_event

    def _on_window_event(self, e: ft.WindowEvent) -> None:
        if e.type == ft.WindowEventType.CLOSE:
            self.shutdown()

    def _on_keyboard_event(self, e: ft.KeyboardEvent) -> None:
        if e.key == "Escape":
            self.shutdown()
        elif e.key == "Arrow Right" or e.key == " ":
            self._next()
        elif e.key == "Arrow Left":
            self._prev()
        elif e.key == "r" or e.key == "R":
            self._reset_games()

    def shutdown(self) -> None:
        if self._shutting_down: return
        self._shutting_down = True
        try: self.cap.release()
        except Exception: pass
        self.page.window.visible = False
        self.page.update()
        time.sleep(SHUTDOWN_GRACE_S)
        kill_process_tree()
        os._exit(0)

    def _next(self) -> None:
        if self.index < len(PAGES) - 1:
            self.index += 1
            self._reset_games()
            self._render()
        else:
            self.shutdown()

    def _prev(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._reset_games()
            self._render()

    def _dot(self, i: int) -> ft.Container:
        active = i == self.index
        return ft.Container(
            width=10 if active else 8, height=10 if active else 8,
            border_radius=6, bgcolor=PAGES[self.index].accent if active else PLEXUS_GRAY,
        )

    def _bullet_list(self, items: list[str], icon: str, color: str) -> ft.Column:
        return ft.Column(
            spacing=10,
            controls=[
                ft.Row(
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Icon(icon, color=color, size=18),
                        ft.Text(item, color=PLEXUS_WHITE, size=15, expand=True),
                    ],
                ) for item in items
            ],
        )

    def _render(self) -> None:
        p = PAGES[self.index]
        is_last = self.index == len(PAGES) - 1

        card = ft.Container(
            padding=40, border_radius=24,
            bgcolor=ft.Colors.with_opacity(0.55, "#111111"),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.15, PLEXUS_WHITE)),
            content=ft.Column(
                spacing=22,
                controls=[
                    ft.Row(
                        controls=[
                            ft.Container(width=6, height=36, border_radius=3, bgcolor=p.accent),
                            ft.Column(
                                spacing=0,
                                controls=[
                                    ft.Text(p.subtitle.upper(), style=ft.TextStyle(size=12, weight=ft.FontWeight.BOLD, color=p.accent, letter_spacing=2)),
                                    ft.Text(p.game_title, size=26, weight=ft.FontWeight.W_900, color=PLEXUS_WHITE),
                                ],
                            ),
                        ]
                    ),
                    ft.Text(p.objective, size=15, color=ft.Colors.WHITE70),
                    ft.Divider(color=ft.Colors.with_opacity(0.1, PLEXUS_WHITE)),
                    ft.Text("HOW TO PLAY", style=ft.TextStyle(size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE54, letter_spacing=2)),
                    self._bullet_list(p.controls, ft.Icons.PAN_TOOL_ALT, p.accent),
                ],
            ),
        )

        nav = ft.Row(
            alignment=ft.MainAxisAlignment.CENTER, spacing=18,
            controls=[
                ft.OutlinedButton("Back", on_click=lambda e: self._prev(), disabled=self.index == 0, style=ft.ButtonStyle(color=PLEXUS_WHITE)),
                ft.Row(spacing=8, controls=[self._dot(i) for i in range(len(PAGES))]),
                ft.FilledButton(
                    "Finish" if is_last else "Next",
                    on_click=lambda e: self._next(),
                    style=ft.ButtonStyle(bgcolor=p.accent, color=PLEXUS_WHITE),
                ),
            ],
        )

        self.left_column.controls = [card, nav]
        self.video_container.border = ft.Border.all(4, p.accent)
        self.page.update()

def main(page: ft.Page) -> None:
    TutorialApp(page)

if __name__ == "__main__":
    ft.run(main)
