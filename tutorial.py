"""
Plexus AI Arcade - Interactive How to Play
A self-contained Flet walkthrough explaining and testing the controls for each game.
Embeds a live OpenCV feed directly into the Flet UI.
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
        game_title="Welcome",
        subtitle="How the Arcade Works",
        objective=(
            "Every station in the Plexus AI Arcade is controlled with your "
            "hand in front of the webcam -- no mouse, no controller. "
            "This walkthrough covers the controls for each game."
        ),
        controls=[
            "Stand where your hand is clearly visible to the camera",
            "Keep your hand inside the center of the frame for best tracking",
            "Try moving your hand now to see the tracking skeleton follow you",
        ],
        shortcuts=["Esc closes the current game and returns to the arcade hub"],
        accent=PLEXUS_GRAY,
    ),
    TutorialPage(
        game_title="SMT Pick and Place",
        subtitle="Circuit Builder",
        objective="Practice the Pinch mechanism to grab components.",
        controls=[
            "Bring your thumb and index finger together to PINCH",
            "Pinch the floating box on the right to turn it green",
            "Move your hand to drag it into the target zone",
        ],
        shortcuts=["R resets the board", "Esc quits to the hub"],
        accent=PLEXUS_RED,
    ),
    TutorialPage(
        game_title="SMT Reflow",
        subtitle="Factory Navigator",
        objective="Practice steering by moving your body left and right.",
        controls=[
            "Move your hand / lean your body LEFT to steer left",
            "Lean RIGHT to steer right",
            "Watch the live gauge to see how your balance controls the lane",
        ],
        shortcuts=["Space starts the run", "Esc quits to the hub"],
        accent="#F39C12",
    ),
    TutorialPage(
        game_title="AOI Inspection",
        subtitle="AOI Inspector",
        objective="Practice the gesture lock mechanism.",
        controls=[
            "Show a clear THUMBS UP gesture",
            "Hold it steady -- watch the progress bar fill up",
            "In-game, this locks in your quality control decisions",
        ],
        shortcuts=["Space starts the round", "Esc quits to the hub"],
        accent=PLEXUS_GREEN,
    ),
    TutorialPage(
        game_title="X-Ray Inspection",
        subtitle="Defect Detective",
        objective="Sweep the X-ray lens across the board.",
        controls=[
            "Move your hand to steer the circular X-ray lens",
            "Pinch your fingers to flag a defect under the lens",
            "Try moving the lens around the screen now",
        ],
        shortcuts=["R resets the board", "Esc quits to the hub"],
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


        # Live Video Feed Control
        self.video_image = ft.Image(
            src=None,
            fit=ft.BoxFit.CONTAIN,
            expand=True,
            gapless_playback=True,   # keeps the last frame visible while the next decodes — reduces flicker
        )

        # Vision State
        self.cursor_x, self.cursor_y = 320.0, 240.0
        self.comp_x, self.comp_y = 150, 240
        self.is_grabbed = False
        self.gesture_frames = 0
        self.CONFIRM_FRAMES = 25

        # Initialize Mediapipe
        self._init_mediapipe()

        # Build Static UI Layout
        self._configure_window()
        self.left_column = ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,  # card/nav fill the wrapper's width
            spacing=30,
        )
        self.left_wrapper = ft.Container(
            expand=4,
            content=self.left_column,
            alignment=ft.Alignment.CENTER,
        )

        self.video_container = ft.Container(
            expand=6,
            border_radius=24,
            border=ft.Border.all(4, PLEXUS_RED),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            content=self.video_image,
            alignment=ft.Alignment.CENTER,
        )

        self.body = ft.Row(
            expand=True,
            spacing=30,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[self.left_wrapper, self.video_container],
        )

        self.page.add(
            ft.Container(
                expand=True,
                padding=50,
                alignment=ft.Alignment(0, 0),
                content=self.body,
            )
        )

        self._render()

        # Camera setup (opened synchronously here; reading happens off-loop)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._video_start_time = time.time()
        self._last_ts = -1

        # Run the capture/render loop as an asyncio task on Flet's own loop.
        # Blocking work (cv2 + mediapipe) is pushed into a worker thread per
        # frame via asyncio.to_thread; only the final `update_async()` touches
        # the UI, and it does so from the loop thread itself.
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

    # -- video processing -------------------------------------------------

    def _capture_and_process(self) -> str | None:
        """
        Runs entirely in a worker thread (via asyncio.to_thread). Does the
        blocking camera read, mediapipe inference, drawing, and JPEG encode.
        Must NOT touch any Flet control or call page/control update methods --
        it only returns a data URI string (or None on failure).
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        is_pinching = False
        pinch_ratio = 1.0

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = int((time.time() - self._video_start_time) * 1000)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts

            hand_res = self.hand_detector.detect_for_video(mp_img, ts)
            gest_res = self.gest_detector.recognize_for_video(mp_img, ts)

            idx = self.index

            if hand_res and hand_res.hand_landmarks:
                lms = hand_res.hand_landmarks[0]

                target_x = max(0, min(w, lms[9].x * w))
                target_y = max(0, min(h, lms[9].y * h))
                self.cursor_x = lerp(self.cursor_x, target_x, 0.4)
                self.cursor_y = lerp(self.cursor_y, target_y, 0.4)

                hand_scale = max(math.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y), 1e-4)
                pinch_ratio = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y) / hand_scale
                is_pinching = pinch_ratio < 0.35

                # PAGE 0: SKELETON
                if idx == 0:
                    for s, e in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                                 (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
                                 (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]:
                        cx1, cy1 = int(lms[s].x * w), int(lms[s].y * h)
                        cx2, cy2 = int(lms[e].x * w), int(lms[e].y * h)
                        cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)
                        cv2.circle(frame, (cx1, cy1), 4, (0, 215, 255), -1)

                # PAGE 2: LEAN
                elif idx == 2:
                    lean_val = (lms[0].x - 0.5) * 2.0
                    cv2.line(frame, (w // 2, 0), (w // 2, h), (100, 100, 100), 2)
                    bar_w = int(lean_val * 200)
                    b_color = (0, 255, 128) if abs(lean_val) > 0.3 else (0, 165, 255)
                    cv2.rectangle(frame, (w // 2, h - 60), (w // 2 + bar_w, h - 30), b_color, -1)
                    cv2.putText(frame, "LEAN GAUGE", (w // 2 - 60, h - 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Overlays that render even if hand is lost
            if idx == 1:
                target_bx = w - 150
                cv2.circle(frame, (target_bx, int(h / 2)), 60, (0, 255, 128), 2)

                if is_pinching and math.hypot(self.cursor_x - self.comp_x, self.cursor_y - self.comp_y) < 50:
                    self.is_grabbed = True
                elif pinch_ratio > 0.55:
                    self.is_grabbed = False

                if self.is_grabbed:
                    self.comp_x, self.comp_y = int(self.cursor_x), int(self.cursor_y)

                c_color = (0, 255, 0) if self.is_grabbed else (0, 165, 255)
                cv2.rectangle(frame, (self.comp_x - 30, self.comp_y - 30),
                              (self.comp_x + 30, self.comp_y + 30), c_color, -1)
                cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 10, (255, 255, 255), 2)

            elif idx == 3:
                gest = "None"
                if gest_res and gest_res.gestures:
                    gest = gest_res.gestures[0][0].category_name

                if gest == "Thumb_Up":
                    self.gesture_frames = min(self.CONFIRM_FRAMES, self.gesture_frames + 1)
                else:
                    self.gesture_frames = max(0, self.gesture_frames - 1)

                cv2.circle(frame, (w // 2, h // 2), 60, (50, 50, 50), 6)
                ang = int((self.gesture_frames / self.CONFIRM_FRAMES) * 360)
                if ang > 0:
                    cv2.ellipse(frame, (w // 2, h // 2), (60, 60), 0, -90, -90 + ang, (0, 255, 128), 6)
                cv2.putText(frame, "THUMBS UP TO FILL", (w // 2 - 80, h // 2 + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif idx == 4:
                mask = np.zeros_like(frame)
                cv2.circle(mask, (int(self.cursor_x), int(self.cursor_y)), 80, (0, 0, 150), -1)
                frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

                l_color = (0, 255, 0) if is_pinching else (0, 215, 255)
                cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 80, l_color, 4)
                cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 5, l_color, -1)

        except Exception as e:
            print(f"Vision error: {e}")

        try:
            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return None
            return buffer.tobytes()
        except Exception as e:
            print(f"Encode error: {e}")
            return None

    async def _video_loop(self):
        """
        Lives on Flet's asyncio loop. Delegates heavy per-frame work to a
        worker thread so the loop stays free to service keyboard events and
        page.update() calls from _render(), then applies the result with a
        scoped update_async() on just the image control.
        """
        while not self._shutting_down:
            loop_start = time.time()

            frame_bytes = await asyncio.to_thread(self._capture_and_process)

            if frame_bytes is not None and not self._shutting_down:
                self.video_image.src = frame_bytes
                try:
                    self.video_image.update()
                except Exception as e:
                    print(f"UI update error: {e}")

            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0.0, FRAME_INTERVAL_S - elapsed))

    # -- lifecycle ------------------------------------------------------

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
        #page.scroll = ft.ScrollMode.AUTO

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

    # -- navigation -------------------------------------------------------

    def _next(self) -> None:
        if self.index < len(PAGES) - 1:
            self.index += 1
            self._render()
        else:
            self.shutdown()

    def _prev(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._render()

    # -- rendering --------------------------------------------------------

    def _dot(self, i: int) -> ft.Container:
        active = i == self.index
        return ft.Container(
            width=10 if active else 8,
            height=10 if active else 8,
            border_radius=6,
            bgcolor=PAGES[self.index].accent if active else PLEXUS_GRAY,
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
                )
                for item in items
            ],
        )

    def _render(self) -> None:
        p = PAGES[self.index]
        is_last = self.index == len(PAGES) - 1

        card = ft.Container(
            padding=40,
            border_radius=24,
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
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=18,
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

        # Update ONLY the left column controls and the video border.
        # The video container remains untouched so the stream never breaks!
        self.left_column.controls = [card, nav]
        self.video_container.border = ft.Border.all(4, p.accent)
        self.page.update()


def main(page: ft.Page) -> None:
    TutorialApp(page)


if __name__ == "__main__":
    ft.run(main)
