"""
Plexus AI Arcade - Interactive How to Play (Authentic Mechanics Edition)
A self-contained Flet walkthrough that mirrors the real game logic loops
wherever practical, teaching the player the precise computer vision
interactions required by each station.
"""

from __future__ import annotations

import os
import time
import base64
import asyncio
import math
import random
import threading
from dataclasses import dataclass, field

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
FRAME_INTERVAL_S = 0.03       # ~33fps target cadence between processed frames
CAMERA_RETRY_INTERVAL_S = 1.5  # how often to retry opening the camera if it's unavailable

PLEXUS_RED = "#DA291C"
PLEXUS_CHARCOAL = "#1A1A1A"
PLEXUS_GRAY = "#333333"
PLEXUS_WHITE = "#FFFFFF"
PLEXUS_GREEN = "#2ECC71"

# --- Defect Detective (shared with DefectDetectiveGame) ---
PINCH_ENGAGE_RATIO = 0.35    # pinch-distance / hand-scale ratio below which a pinch counts as "closed"
PINCH_RELEASE_RATIO = 0.55   # ratio above which a previously-closed pinch counts as "released"
LENS_RADIUS_PX = 80          # kept as originally tuned for the tutorial's smaller frame
DD_DEFECT_COUNT = 4          # randomized each time this page is (re)loaded

# --- Circuit Builder (matches CircuitBuilderGame exactly) ---
CB_PINCH_GRAB_RATIO = 0.20
CB_PINCH_DROP_RATIO = 0.45
SOCKET_SNAP_RADIUS_PX = 60

# --- Factory Navigator (matches FactoryNavigatorGame exactly) ---
FN_LEAN_RATIO_THRESHOLD = 0.35
FN_ROBOT_SPEED = 6            # px/frame nudge applied while leaning past the threshold
STEER_LANE_TOLERANCE_PX = 50

# Model files are resolved relative to this script, not the process's current
# working directory, so the app finds them regardless of how it's launched.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'hand_landmarker.task')
GESTURE_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'gesture_recognizer.task')
POSE_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'pose_landmarker_lite.task')  # matches FactoryNavigatorGame


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
    # General, non-sequential guidance: (text, icon) pairs, each with an icon
    # matched to the action it describes rather than one icon for everything.
    tips: list[tuple[str, str]]
    # Ordered steps that map 1:1 onto this page's internal stage machine
    # (empty for pages with no meaningful stage order). Rendered as numbered
    # badges that highlight live as the player progresses.
    steps: list[str] = field(default_factory=list)
    shortcuts: list[str] = field(default_factory=list)
    accent: str = PLEXUS_RED
    # Which detectors this page actually needs. Gating inference by page
    # avoids running the (relatively expensive) pose model on pages that
    # never look at body landmarks, etc.
    needs_hand: bool = False
    needs_gesture: bool = False
    needs_pose: bool = False
    # True only for Circuit Builder: that game's real hand detector runs in
    # single-shot IMAGE mode rather than VIDEO mode, so it needs its own
    # detector instance and call signature.
    uses_image_mode_hand: bool = False


PAGES: list[TutorialPage] = [
    TutorialPage(
        game_title="System Calibration",
        subtitle="Operator Setup",
        objective=(
            "Every station in the Plexus AI Arcade tracks your body in 3D space. "
            "Ensure the system can see you clearly before starting."
        ),
        tips=[
            ("Stand about arm's length from the camera", ft.Icons.DIRECTIONS_WALK),
            ("Keep your hand inside the frame with fingers spread", ft.Icons.PAN_TOOL_ALT),
            ("Verify both the hand and upper-body skeletons map cleanly to your joints", ft.Icons.GRADING),
        ],
        shortcuts=["Esc closes the current game and returns to the arcade hub"],
        accent=PLEXUS_GRAY,
        needs_hand=True,
        needs_pose=True,
    ),
    TutorialPage(
        game_title="SMT Pick and Place",
        subtitle="Circuit Builder",
        objective="Learn the dynamic pinch scaling calculation used to manipulate components.",
        tips=[
            ("Keep your palm open facing the camera", ft.Icons.BACK_HAND),
            ("Pinch using ONLY your thumb and index finger", ft.Icons.TOUCH_APP),
            ("Maintain your other three fingers OPEN so the camera tracks the pinch accurately", ft.Icons.PAN_TOOL_ALT),
        ],
        steps=[
            "Grab component 1 and place it in Socket 1",
            "Grab component 2 and place it in Socket 2",
            "Grab component 3 and place it in Socket 3",
        ],
        shortcuts=["R resets the sequence if a component gets stuck"],
        accent=PLEXUS_RED,
        needs_hand=True,
        uses_image_mode_hand=True,
    ),
    TutorialPage(
        game_title="SMT Reflow",
        subtitle="Factory Navigator",
        objective="Learn the shoulder-tilt steering used to guide the PCB down the conveyor.",
        tips=[
            ("Stand back so both shoulders are clearly visible to the camera", ft.Icons.ACCESSIBILITY_NEW),
            ("The system measures lean by comparing the height of your left and right shoulder", ft.Icons.GPS_FIXED),
        ],
        steps=[
            "Keep your hips planted and lean your torso LEFT",
            "Now lean your torso RIGHT",
        ],
        shortcuts=["Space starts the run in the actual game"],
        accent="#F39C12",
        needs_pose=True,
    ),
    TutorialPage(
        game_title="AOI Inspection",
        subtitle="AOI Inspector",
        objective="Learn the frame-debouncing logic used to prevent accidental decisions.",
        tips=[
            ("Hold the gesture perfectly still to fill the confidence ring", ft.Icons.GRADING),
        ],
        steps=[
            "Make a clear 'Thumbs Up' gesture to approve the board",
            "Make a clear 'Thumbs Down' gesture to reject the board",
        ],
        shortcuts=["The system requires 35 consecutive frames to lock a decision"],
        accent=PLEXUS_GREEN,
        needs_hand=True,
        needs_gesture=True,
    ),
    TutorialPage(
        game_title="X-Ray Inspection",
        subtitle="Defect Detective",
        objective="Combine steering and pinching to operate the X-ray lens.",
        tips=[
            ("Move your open hand to steer the X-ray lens across the board", ft.Icons.PAN_TOOL_ALT),
            ("When you spot a red hidden defect, hover the lens over it", ft.Icons.SEARCH),
            ("Perform a sharp pinch gesture to successfully flag the defect", ft.Icons.TOUCH_APP),
        ],
        shortcuts=["Find all hidden defects to pass the actual game"],
        accent="#3498DB",
        needs_hand=True,
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
        self._init_error: str | None = None

        # Persistent references to the live-updating progress UI. Populated
        # by _render()/_build_progress_panel() and mutated in place every
        # frame by _refresh_progress_ui(), rather than rebuilding the whole
        # card each frame.
        self.progress_panel: ft.Control | None = None
        self.progress_bar: ft.ProgressBar | None = None
        self.progress_label: ft.Text | None = None
        self.step_controls: list[tuple[ft.Container, ft.Text, ft.Text]] = []

        # A single lock protects all of the mutable per-game state below,
        # since it's written by the worker thread that processes each video
        # frame (_capture_and_process) while also being reset from the main
        # loop thread via keyboard/button-driven navigation (_reset_games),
        # and read from the main loop thread for progress-panel updates.
        self._state_lock = threading.Lock()

        # A tiny placeholder frame shown before the camera has produced its
        # first real frame, so the video area is never blank/None.
        self.video_image = ft.Image(
            src=self._make_message_frame(["STARTING CAMERA..."], _hex_to_bgr(PLEXUS_GRAY)),
            fit=ft.BoxFit.CONTAIN, expand=True, gapless_playback=True,
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
        self.cb_pinch_active = False   # hysteresis-based pinch gesture state (matches CircuitBuilderGame.is_pinching)
        self.cb_is_grabbed = False     # whether a component is currently being dragged
        self.cb_success = False

        # Game 2: Factory Navigator (Left then Right)
        self.fn_stage = 0  # 0 = Left, 1 = Right
        self.fn_lean_ratio = 0.0
        self.fn_target_x = 320.0
        self.fn_pcb_x = 320.0
        self.fn_steering_label = "CENTERED"
        self.fn_success = False

        # Game 3: AOI Inspector (Up then Down)
        self.aoi_frames = 0
        self.AOI_CONFIRM_FRAMES = 35
        self.aoi_stage = 0  # 0 = Thumbs Up, 1 = Thumbs Down
        self.aoi_success = False

        # Game 4: Defect Detective (randomized multi-defect sweep)
        self.dd_defects = self._spawn_dd_defects()
        self.dd_success = False

        # Configure the window and show a loading screen immediately, since
        # model loading below is a blocking call that can take a moment --
        # better than a blank window while it happens.
        self._configure_window()
        self._show_loading_screen()

        # Model loading can fail (missing/corrupt .task files); fail gracefully
        # with an on-screen message instead of an unhandled traceback.
        self._init_error = self._init_mediapipe()
        if self._init_error:
            self._show_fatal_error(self._init_error)
            return

        self.page.controls.clear()
        self._build_ui()

        # Camera setup. If it's not available at startup, the video loop will
        # keep retrying periodically and show a clear on-screen message
        # instead of a frozen/blank feed.
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._video_start_time = time.time()
        self._last_ts = -1
        self.page.run_task(self._video_loop)

    def _init_mediapipe(self) -> str | None:
        """Loads all Mediapipe models. Returns an error message string on
        failure, or None on success."""
        try:
            # Shared VIDEO-mode hand landmarker, used by Calibration, AOI, and
            # Defect Detective (none of which were asked to match a specific
            # game's exact hand-model configuration).
            base_opt_h = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
            self.hand_detector = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(base_options=base_opt_h, num_hands=1, running_mode=vision.RunningMode.VIDEO)
            )

            # Circuit Builder gets its own dedicated hand landmarker matching
            # CircuitBuilderGame exactly: single-shot IMAGE mode on CPU.
            base_opt_cb = python.BaseOptions(model_asset_path=HAND_MODEL_PATH, delegate=python.BaseOptions.Delegate.CPU)
            self.cb_hand_detector = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(base_options=base_opt_cb, num_hands=1, running_mode=vision.RunningMode.IMAGE)
            )

            base_opt_g = python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH)
            self.gest_detector = vision.GestureRecognizer.create_from_options(
                vision.GestureRecognizerOptions(base_options=base_opt_g, num_hands=1, running_mode=vision.RunningMode.VIDEO)
            )

            base_opt_p = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
            self.pose_detector = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(base_options=base_opt_p, output_segmentation_masks=False, running_mode=vision.RunningMode.VIDEO)
            )
            return None
        except Exception as e:
            return (
                "Failed to load a required AI model file. Make sure "
                "hand_landmarker.task, gesture_recognizer.task, and "
                f"pose_landmarker_lite.task are next to this script.\n\nDetails: {e}"
            )

    def _show_loading_screen(self) -> None:
        self.page.add(
            ft.Container(
                expand=True,
                alignment=ft.Alignment(0, 0),
                padding=50,
                content=ft.Column(
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=18,
                    controls=[
                        ft.ProgressRing(color=PLEXUS_RED, width=42, height=42, stroke_width=4),
                        ft.Text("LOADING AI MODELS...", size=14, weight=ft.FontWeight.BOLD, color=PLEXUS_WHITE),
                    ],
                ),
            )
        )
        self.page.update()

    def _show_fatal_error(self, message: str) -> None:
        """Displays a startup error in place of the normal UI, instead of
        letting the app crash with a raw traceback."""
        self.page.controls.clear()
        self.page.add(
            ft.Container(
                expand=True,
                alignment=ft.Alignment(0, 0),
                padding=50,
                content=ft.Container(
                    padding=40,
                    border_radius=24,
                    bgcolor=ft.Colors.with_opacity(0.7, "#111111"),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.3, PLEXUS_RED)),
                    content=ft.Column(
                        spacing=16,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(ft.Icons.ERROR_OUTLINE, color=PLEXUS_RED, size=48),
                            ft.Text("Startup Error", size=22, weight=ft.FontWeight.W_900, color=PLEXUS_WHITE),
                            ft.Text(message, size=14, color=ft.Colors.WHITE70, text_align=ft.TextAlign.CENTER),
                        ],
                    ),
                ),
            )
        )
        self.page.update()

    def _build_ui(self):
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

    def _spawn_dd_defects(self, w: int = 640, h: int = 480, count: int = DD_DEFECT_COUNT) -> list[dict]:
        """Randomly places `count` defects, spaced apart, within the frame."""
        margin = 90
        min_spacing = 100
        pts: list[tuple[int, int]] = []
        attempts = 0
        while len(pts) < count and attempts < 200:
            attempts += 1
            x = random.randint(margin, w - margin)
            y = random.randint(margin, h - margin)
            if all(math.hypot(x - px, y - py) > min_spacing for px, py in pts):
                pts.append((x, y))
        while len(pts) < count:  # fallback if spacing couldn't be satisfied
            pts.append((random.randint(margin, w - margin), random.randint(margin, h - margin)))
        return [{"x": x, "y": y, "found": False} for x, y in pts]

    def _reset_games(self):
        """Resets all minigame state logic. Locked because this can run on
        the main thread (nav/keyboard) while a video frame is mid-processing
        on a worker thread."""
        with self._state_lock:
            self.cb_stage = 0
            self.cb_comp_x, self.cb_comp_y = self.cb_stages[0][:2]
            self.cb_pinch_active = False
            self.cb_is_grabbed = False
            self.cb_success = False

            self.fn_stage = 0
            self.fn_lean_ratio = 0.0
            self.fn_target_x = 320.0
            self.fn_pcb_x = 320.0
            self.fn_steering_label = "CENTERED"
            self.fn_success = False

            self.aoi_stage = 0
            self.aoi_frames = 0
            self.aoi_success = False

            self.dd_defects = self._spawn_dd_defects()
            self.dd_success = False

    def _is_stage_complete(self, i: int) -> bool:
        return {
            1: self.cb_success,
            2: self.fn_success,
            3: self.aoi_success,
            4: self.dd_success,
        }.get(i, False)

    def _progress_for_page(self, idx: int) -> tuple[float, str, int | None]:
        """Returns (fraction 0..1, status label, current_step_index or None)
        describing live progress for the given page's minigame."""
        with self._state_lock:
            if idx == 1:
                total = len(self.cb_stages)
                if self.cb_success:
                    return 1.0, f"All {total} components placed", total
                return self.cb_stage / total, f"Socket {self.cb_stage + 1} of {total}", self.cb_stage
            if idx == 2:
                total = 2
                if self.fn_success:
                    return 1.0, "Lean calibration complete", total
                label = "Lean left to continue" if self.fn_stage == 0 else "Lean right to continue"
                return self.fn_stage / total, label, self.fn_stage
            if idx == 3:
                total = 2
                stage_frac = self.aoi_frames / self.AOI_CONFIRM_FRAMES
                if self.aoi_success:
                    return 1.0, "Both decisions locked", total
                gesture_name = "thumbs up" if self.aoi_stage == 0 else "thumbs down"
                return (self.aoi_stage + stage_frac) / total, f"Holding {gesture_name} ({int(stage_frac * 100)}%)", self.aoi_stage
            if idx == 4:
                total = len(self.dd_defects)
                found = sum(1 for d in self.dd_defects if d["found"])
                if self.dd_success:
                    return 1.0, f"All {total} defects flagged", None
                return (found / total if total else 0.0), f"{found} of {total} defects found", None
        return 0.0, "", None

    def _refresh_progress_ui(self, idx: int) -> None:
        """Runs on the Flet event-loop thread once per processed video frame,
        mutating the persistent progress controls in place rather than
        rebuilding the card. No-op on pages with no progress panel (e.g. the
        calibration page) or if the page has since been navigated away from."""
        if idx != self.index or self.progress_panel is None:
            return

        fraction, label, current_step = self._progress_for_page(idx)
        self.progress_bar.value = fraction
        self.progress_label.value = label

        for i, (badge, badge_text, step_label) in enumerate(self.step_controls):
            if current_step is not None and i < current_step:
                badge.bgcolor = PLEXUS_GREEN
                badge_text.value = "\u2713"
                step_label.color = ft.Colors.WHITE70
            elif current_step is not None and i == current_step:
                badge.bgcolor = PAGES[idx].accent
                badge_text.value = str(i + 1)
                step_label.color = PLEXUS_WHITE
            else:
                badge.bgcolor = PLEXUS_GRAY
                badge_text.value = str(i + 1)
                step_label.color = ft.Colors.WHITE38

        try:
            self.progress_panel.update()
        except Exception:
            pass

    # -- camera handling ----------------------------------------------------

    def _make_message_frame(self, lines: list[str], accent_bgr: tuple[int, int, int]) -> str:
        """Builds a placeholder frame (as a base64 data URI) carrying a
        short message, used when the camera is unavailable or hasn't
        produced its first frame yet."""
        w, h = 640, 480
        frame = np.full((h, w, 3), 20, dtype=np.uint8)
        cv2.rectangle(frame, (4, 4), (w - 5, h - 5), accent_bgr, 8)
        y = h // 2 - (len(lines) - 1) * 18
        for line in lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x = (w - text_size[0]) // 2
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 36
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_str = base64.b64encode(buffer).decode('utf-8') if ok else ""
        return f"data:image/jpeg;base64,{b64_str}"

    def _handle_camera_unavailable(self) -> str | None:
        """Runs in a worker thread. Attempts to reopen the camera; if that
        succeeds, returns None so the next loop iteration resumes normal
        frame capture. Otherwise returns a placeholder message frame."""
        if self.cap.open(0):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return None
        accent_bgr = _hex_to_bgr(PAGES[self.index].accent)
        return self._make_message_frame(
            ["NO CAMERA DETECTED", "Check the connection and try again"], accent_bgr
        )

    # -- video processing -----------------------------------------------------

    def _capture_and_process(self) -> str | None:
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        is_pinching = False
        pinch_ratio = 1.0
        hand_detected = False
        pose_detected = False

        idx = self.index
        page_cfg = PAGES[idx]
        accent_bgr = _hex_to_bgr(page_cfg.accent)

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.time() - self._video_start_time) * 1000)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts

            # Only run the detectors the current page actually needs -- the
            # pose model in particular is comparatively expensive, so this
            # keeps frame latency down on pages that don't use it.
            hand_res = None
            if page_cfg.needs_hand:
                if page_cfg.uses_image_mode_hand:
                    # Matches CircuitBuilderGame: single-shot IMAGE-mode detect().
                    hand_res = self.cb_hand_detector.detect(mp_img)
                else:
                    hand_res = self.hand_detector.detect_for_video(mp_img, ts)
            gest_res = self.gest_detector.recognize_for_video(mp_img, ts) if page_cfg.needs_gesture else None
            pose_res = self.pose_detector.detect_for_video(mp_img, ts) if page_cfg.needs_pose else None

            # --- 1. BODY/POSE TRACKING ---
            if pose_res and pose_res.pose_landmarks:
                pose_detected = True
                pose_lms = pose_res.pose_landmarks[0]
                left_shoulder = pose_lms[11]
                right_shoulder = pose_lms[12]

                # Factory Navigator's authentic shoulder-tilt lean metric
                # (matches FactoryNavigatorGame exactly -- shoulders only, no hips).
                shoulder_width = max(abs(left_shoulder.x - right_shoulder.x), 1e-4)
                raw_lean_diff = left_shoulder.y - right_shoulder.y
                self.fn_lean_ratio = raw_lean_diff / shoulder_width

                if idx == 0:
                    # Calibration keeps its own richer upper-body visualization,
                    # since this page isn't matching a specific game's mechanic.
                    mid_hip_x = (pose_lms[23].x + pose_lms[24].x) / 2.0
                    mid_hip_y = (pose_lms[23].y + pose_lms[24].y) / 2.0
                    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
                    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0

                    for s, e in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]:
                        cx1, cy1 = int(pose_lms[s].x * w), int(pose_lms[s].y * h)
                        cx2, cy2 = int(pose_lms[e].x * w), int(pose_lms[e].y * h)
                        cv2.line(frame, (cx1, cy1), (cx2, cy2), accent_bgr, 2)
                        cv2.circle(frame, (cx1, cy1), 4, (255, 255, 255), -1)
                        cv2.circle(frame, (cx2, cy2), 4, (255, 255, 255), -1)

                    sx, sy = int(mid_shoulder_x * w), int(mid_shoulder_y * h)
                    hx, hy = int(mid_hip_x * w), int(mid_hip_y * h)
                    cv2.line(frame, (hx, hy), (sx, sy), (0, 255, 128), 5)
                    cv2.circle(frame, (sx, sy), 8, (0, 255, 128), -1)

                elif idx == 2:
                    # Matches FactoryNavigatorGame's PiP: a simple shoulder line
                    # that changes color once it crosses the lean threshold.
                    lx, ly = int(left_shoulder.x * w), int(left_shoulder.y * h)
                    rx, ry = int(right_shoulder.x * w), int(right_shoulder.y * h)
                    lean_color = (0, 255, 128) if abs(self.fn_lean_ratio) > FN_LEAN_RATIO_THRESHOLD else accent_bgr
                    cv2.line(frame, (lx, ly), (rx, ry), lean_color, 4)
                    cv2.circle(frame, (lx, ly), 7, lean_color, -1)
                    cv2.circle(frame, (rx, ry), 7, lean_color, -1)

            # --- 2. HAND TRACKING ---
            if hand_res and hand_res.hand_landmarks:
                hand_detected = True
                lms = hand_res.hand_landmarks[0]

                target_x = max(0, min(w, lms[9].x * w))
                target_y = max(0, min(h, lms[9].y * h))
                self.cursor_x = lerp(self.cursor_x, target_x, 0.4)
                self.cursor_y = lerp(self.cursor_y, target_y, 0.4)

                hand_scale = max(math.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y), 1e-4)
                pinch_ratio = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y) / hand_scale
                is_pinching = pinch_ratio < PINCH_ENGAGE_RATIO

                if idx in (0, 1, 3, 4):
                    for s, e in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                                 (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
                                 (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]:
                        cx1, cy1 = int(lms[s].x * w), int(lms[s].y * h)
                        cx2, cy2 = int(lms[e].x * w), int(lms[e].y * h)
                        cv2.line(frame, (cx1, cy1), (cx2, cy2), accent_bgr, 2)
                        cv2.circle(frame, (cx1, cy1), 4, (255, 255, 255), -1)

            # --- RENDER TUTORIAL STATES ---
            # All per-game state mutation is locked, since it's shared with
            # the main-thread reset/nav/progress-panel handlers.
            with self._state_lock:

                # STATE 0: Welcome Calibration
                if idx == 0:
                    if hand_detected and pose_detected:
                        cv2.putText(frame, "STATUS: TRACKING ACTIVE", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
                    else:
                        missing = []
                        if not hand_detected:
                            missing.append("HAND")
                        if not pose_detected:
                            missing.append("BODY")
                        cv2.putText(frame, f"STATUS: LOCATING {' & '.join(missing)}...", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # STATE 1: Circuit Builder Logic (Multi-stage)
                elif idx == 1:
                    # Authentic hysteresis-based pinch state (grab below
                    # CB_PINCH_GRAB_RATIO, release above CB_PINCH_DROP_RATIO),
                    # matching CircuitBuilderGame.is_pinching exactly.
                    if not self.cb_pinch_active and pinch_ratio < CB_PINCH_GRAB_RATIO:
                        self.cb_pinch_active = True
                    elif self.cb_pinch_active and pinch_ratio > CB_PINCH_DROP_RATIO:
                        self.cb_pinch_active = False

                    if self.cb_success:
                        tx, ty, tcolor = self.cb_stages[-1][2], self.cb_stages[-1][3], self.cb_stages[-1][4]
                    else:
                        tx, ty, tcolor = self.cb_stages[self.cb_stage][2], self.cb_stages[self.cb_stage][3], self.cb_stages[self.cb_stage][4]

                    cv2.circle(frame, (tx, ty), 50, accent_bgr, 3)
                    cv2.putText(frame, f"SOCKET {self.cb_stage + 1}/{len(self.cb_stages)}", (tx - 40, ty - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, accent_bgr, 2)

                    # Grab detection: rectangle hit-test against the component's
                    # bounding box, matching CircuitBuilderGame's Component rect
                    # check (not a radius/distance check).
                    comp_left, comp_right = self.cb_comp_x - 30, self.cb_comp_x + 30
                    comp_top, comp_bottom = self.cb_comp_y - 30, self.cb_comp_y + 30

                    if not self.cb_is_grabbed and self.cb_pinch_active and not self.cb_success:
                        if comp_left < self.cursor_x < comp_right and comp_top < self.cursor_y < comp_bottom:
                            self.cb_is_grabbed = True
                    elif self.cb_is_grabbed and not self.cb_pinch_active:
                        self.cb_is_grabbed = False
                        if math.hypot(self.cb_comp_x - tx, self.cb_comp_y - ty) < SOCKET_SNAP_RADIUS_PX:
                            if self.cb_stage < len(self.cb_stages) - 1:
                                self.cb_stage += 1
                                self.cb_comp_x, self.cb_comp_y = self.cb_stages[self.cb_stage][:2]
                            else:
                                self.cb_success = True
                                self.cb_comp_x, self.cb_comp_y = tx, ty  # Snap to final socket

                    if self.cb_is_grabbed:
                        self.cb_comp_x, self.cb_comp_y = int(self.cursor_x), int(self.cursor_y)

                    c_color = (0, 255, 0) if self.cb_success else ((0, 255, 255) if self.cb_is_grabbed else tcolor)
                    cv2.rectangle(frame, (self.cb_comp_x - 30, self.cb_comp_y - 30), (self.cb_comp_x + 30, self.cb_comp_y + 30), c_color, -1)

                    if self.cb_pinch_active:
                        cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 12, (0, 255, 0), cv2.FILLED)
                    else:
                        cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 15, (255, 255, 255), 2)

                    if self.cb_success:
                        cv2.putText(frame, "ALL COMPONENTS PLACED!", (w // 2 - 140, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

                # STATE 2: Factory Navigator Logic (Left -> Right)
                elif idx == 2:
                    # Velocity-based control: nudge a target position while
                    # leaning past the threshold, then ease the visible PCB
                    # toward it -- matches FactoryNavigatorGame's steering
                    # exactly (not a direct position mapping).
                    if self.fn_lean_ratio > FN_LEAN_RATIO_THRESHOLD:
                        self.fn_steering_label = "TURNING RIGHT"
                        steer_color = (0, 255, 128)
                        self.fn_target_x += FN_ROBOT_SPEED
                    elif self.fn_lean_ratio < -FN_LEAN_RATIO_THRESHOLD:
                        self.fn_steering_label = "TURNING LEFT"
                        steer_color = (60, 120, 255)
                        self.fn_target_x -= FN_ROBOT_SPEED
                    else:
                        self.fn_steering_label = "CENTERED"
                        steer_color = (120, 220, 255)

                    self.fn_target_x = max(45.0, min(w - 45.0, self.fn_target_x))
                    self.fn_pcb_x = lerp(self.fn_pcb_x, self.fn_target_x, 0.15)
                    pcb_x = int(self.fn_pcb_x)

                    # Conveyor lines
                    cv2.line(frame, (w // 2 - 120, 0), (w // 2 - 250, h), (100, 100, 100), 2)
                    cv2.line(frame, (w // 2, 0), (w // 2, h), (70, 70, 70), 2)
                    cv2.line(frame, (w // 2 + 120, 0), (w // 2 + 250, h), (100, 100, 100), 2)

                    # Target lane based on stage
                    target_lane_x = w // 2 + (-150 if self.fn_stage == 0 else 150)

                    if not self.fn_success:
                        cv2.rectangle(frame, (target_lane_x - 60, h // 2 + 60), (target_lane_x + 60, h // 2 + 180), accent_bgr, 2)
                        cv2.putText(frame, "STEER HERE", (target_lane_x - 45, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, accent_bgr, 2)

                    # Collision Check
                    if abs(pcb_x - target_lane_x) < STEER_LANE_TOLERANCE_PX:
                        if self.fn_stage == 0:
                            self.fn_stage = 1  # Move to Right stage
                        else:
                            self.fn_success = True

                    # Draw Steering PCB
                    pcb_color = (0, 255, 0) if self.fn_success else (0, 150, 255)
                    cv2.rectangle(frame, (pcb_x - 45, h // 2 + 80), (pcb_x + 45, h // 2 + 160), pcb_color, -1)
                    cv2.rectangle(frame, (pcb_x - 45, h // 2 + 80), (pcb_x + 45, h // 2 + 160), (255, 255, 255), 2)

                    if self.fn_success:
                        cv2.putText(frame, "LEAN CALIBRATION SUCCESS!", (w // 2 - 160, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"STATUS: {self.fn_steering_label}", (w // 2 - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, steer_color, 2)

                # STATE 3: AOI Inspector Logic (Up -> Down)
                elif idx == 3:
                    gest = "None"
                    if gest_res and gest_res.gestures and gest_res.gestures[0]:
                        gest = gest_res.gestures[0][0].category_name

                    target_gest = "Thumb_Up" if self.aoi_stage == 0 else "Thumb_Down"

                    if gest == target_gest:
                        self.aoi_frames = min(self.AOI_CONFIRM_FRAMES, self.aoi_frames + 1)
                    else:
                        self.aoi_frames = max(0, self.aoi_frames - 1)

                    cv2.circle(frame, (w // 2, h // 2), 60, (50, 50, 50), 6)
                    ang = int((self.aoi_frames / self.AOI_CONFIRM_FRAMES) * 360)
                    if ang > 0:
                        cv2.ellipse(frame, (w // 2, h // 2), (60, 60), 0, -90, -90 + ang, accent_bgr, 6)

                    if self.aoi_frames >= self.AOI_CONFIRM_FRAMES:
                        if self.aoi_stage == 0:
                            self.aoi_stage = 1
                            self.aoi_frames = 0
                        else:
                            self.aoi_success = True

                    pct = int((self.aoi_frames / self.AOI_CONFIRM_FRAMES) * 100)
                    if self.aoi_success:
                        cv2.putText(frame, "ALL DECISIONS LOCKED!", (w // 2 - 110, h // 2 + 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 128), 2)
                    else:
                        instruction = f"HOLD THUMBS UP ({pct}%)" if self.aoi_stage == 0 else f"HOLD THUMBS DOWN ({pct}%)"
                        cv2.putText(frame, instruction, (w // 2 - 100, h // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # STATE 4: Defect Detective Logic (randomized multi-defect sweep)
                elif idx == 4:
                    # True "hidden defect" spotlight: everything outside the lens
                    # is dimmed on a copy of the frame that never had the defects
                    # drawn on it, so they're genuinely hidden until the lens
                    # passes over them -- not just visually dimmed.
                    revealed = frame.copy()
                    for d in self.dd_defects:
                        if not d["found"]:
                            cv2.circle(revealed, (d["x"], d["y"]), 15, (0, 0, 255), -1)
                    # Slight red boost inside the lens for an "X-ray" feel.
                    revealed[:, :, 2] = np.clip(revealed[:, :, 2].astype(np.int16) + 30, 0, 255).astype(np.uint8)

                    dark = (frame.astype(np.float32) * 0.25).astype(np.uint8)

                    lens_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(lens_mask, (int(self.cursor_x), int(self.cursor_y)), LENS_RADIUS_PX, 255, -1)
                    frame = np.where(lens_mask[:, :, None] == 255, revealed, dark)

                    if is_pinching:
                        for d in self.dd_defects:
                            if not d["found"] and math.hypot(self.cursor_x - d["x"], self.cursor_y - d["y"]) < LENS_RADIUS_PX:
                                d["found"] = True

                    for d in self.dd_defects:
                        if d["found"]:
                            cv2.rectangle(frame, (d["x"] - 30, d["y"] - 30), (d["x"] + 30, d["y"] + 30), (0, 255, 0), 3)

                    found_count = sum(1 for d in self.dd_defects if d["found"])
                    if found_count == len(self.dd_defects):
                        self.dd_success = True
                        cv2.putText(frame, "ALL DEFECTS FLAGGED!", (w // 2 - 140, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"DEFECTS FOUND: {found_count}/{len(self.dd_defects)}", (w // 2 - 130, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    l_color = (0, 255, 0) if is_pinching else accent_bgr
                    cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), LENS_RADIUS_PX, l_color, 4)
                    cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 5, l_color, -1)

        except Exception as e:
            print(f"Vision error: {e}")

        t = 4
        cv2.rectangle(frame, (t, t), (w - 1 - t, h - 1 - t), accent_bgr, 8)

        try:
            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return None
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception as e:
            print(f"Encode error: {e}")
            return None

    async def _video_loop(self):
        while not self._shutting_down:
            loop_start = time.time()
            current_idx = self.index

            if self.cap.isOpened():
                frame_data_uri = await asyncio.to_thread(self._capture_and_process)
            else:
                frame_data_uri = await asyncio.to_thread(self._handle_camera_unavailable)

            if frame_data_uri is not None and not self._shutting_down:
                self.video_image.src = frame_data_uri
                try:
                    self.video_image.update()
                except Exception:
                    pass
                self._refresh_progress_ui(current_idx)

            elapsed = time.time() - loop_start
            sleep_for = FRAME_INTERVAL_S if self.cap.isOpened() else CAMERA_RETRY_INTERVAL_S
            await asyncio.sleep(max(0.0, sleep_for - elapsed))

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
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            self.cap.release()
        except Exception:
            pass
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
        completed = self._is_stage_complete(i)
        if completed:
            color = PLEXUS_GREEN
        elif active:
            color = PAGES[self.index].accent
        else:
            color = PLEXUS_GRAY
        size = 10 if active else 8
        return ft.Container(
            width=size, height=size,
            border_radius=size // 2 + 1,
            bgcolor=color,
        )

    def _tip_list(self, items: list[tuple[str, str]], color: str) -> ft.Column:
        return ft.Column(
            spacing=10,
            controls=[
                ft.Row(
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Icon(icon, color=color, size=18),
                        ft.Text(text, color=PLEXUS_WHITE, size=15, expand=True),
                    ],
                ) for text, icon in items
            ],
        )

    def _build_progress_panel(self, p: TutorialPage) -> ft.Control | None:
        """Builds (and stores references to) the live progress bar, status
        label, and numbered step badges for pages that have a minigame.
        Returns None for pages with nothing to track (e.g. calibration)."""
        if self.index == 0:
            self.progress_bar = None
            self.progress_label = None
            self.step_controls = []
            self.progress_panel = None
            return None

        self.progress_bar = ft.ProgressBar(
            value=0, color=p.accent, bgcolor=ft.Colors.with_opacity(0.12, PLEXUS_WHITE), border_radius=6,
        )
        self.progress_label = ft.Text("", size=13, color=ft.Colors.WHITE70)

        step_rows = []
        self.step_controls = []
        for i, step_text in enumerate(p.steps):
            badge_text = ft.Text(str(i + 1), size=12, weight=ft.FontWeight.BOLD, color=PLEXUS_WHITE)
            badge = ft.Container(
                width=24, height=24, border_radius=12,
                alignment=ft.Alignment(0, 0), bgcolor=PLEXUS_GRAY, content=badge_text,
            )
            step_label = ft.Text(step_text, size=14, color=ft.Colors.WHITE38, expand=True)
            step_rows.append(
                ft.Row(spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[badge, step_label])
            )
            self.step_controls.append((badge, badge_text, step_label))

        children: list[ft.Control] = [
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.Text("PROGRESS", style=ft.TextStyle(size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE54, letter_spacing=2)),
                    self.progress_label,
                ],
            ),
            self.progress_bar,
        ]
        if step_rows:
            children.append(ft.Column(spacing=10, controls=step_rows))

        self.progress_panel = ft.Column(spacing=14, controls=children)
        return self.progress_panel

    def _render(self) -> None:
        p = PAGES[self.index]
        is_last = self.index == len(PAGES) - 1

        card_children: list[ft.Control] = [
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
            self._tip_list(p.tips, p.accent),
        ]

        progress_panel = self._build_progress_panel(p)
        if progress_panel is not None:
            card_children.append(ft.Divider(color=ft.Colors.with_opacity(0.1, PLEXUS_WHITE)))
            card_children.append(progress_panel)

        card = ft.Container(
            padding=40, border_radius=24,
            bgcolor=ft.Colors.with_opacity(0.55, "#111111"),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.15, PLEXUS_WHITE)),
            content=ft.Column(spacing=22, controls=card_children),
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

        # Populate the freshly built progress panel with the current
        # (just-reset) state immediately, rather than waiting for the next
        # video frame to tick it over.
        self._refresh_progress_ui(self.index)

def main(page: ft.Page) -> None:
    TutorialApp(page)

if __name__ == "__main__":
    ft.run(main)
