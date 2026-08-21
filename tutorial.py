"""
Plexus AI Arcade - How to Play
A self-contained Flet walkthrough explaining the controls for each game.
Meant to be launched as a subprocess from arcade_hub_flet.py, but also
runs standalone: `python tutorial.py`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import flet as ft
import psutil

# --------------------------------------------------------------------------
# Shared visual language (matches arcade_hub_flet.py)
# --------------------------------------------------------------------------

APP_TITLE = "Plexus AI Arcade - How to Play"
SHUTDOWN_GRACE_S = 0.2
CHILD_TERMINATE_TIMEOUT_S = 2.0

PLEXUS_RED = "#DA291C"
PLEXUS_CHARCOAL = "#1A1A1A"
PLEXUS_GRAY = "#333333"
PLEXUS_WHITE = "#FFFFFF"
PLEXUS_GREEN = "#2ECC71"


def kill_process_tree() -> None:
    """Mirrors arcade_hub_flet.py's fix: the Flet renderer is a separate
    process from Python, so exiting Python alone won't close the window."""
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


# --------------------------------------------------------------------------
# Content -- pulled from the actual game source where available.
# --------------------------------------------------------------------------

@dataclass
class TutorialPage:
    game_title: str
    subtitle: str
    objective: str
    controls: list[str]
    shortcuts: list[str]
    accent: str = PLEXUS_RED
    is_placeholder: bool = False


PAGES: list[TutorialPage] = [
    TutorialPage(
        game_title="Welcome",
        subtitle="How the Arcade Works",
        objective=(
            "Every station in the Plexus AI Arcade is controlled with your "
            "hand in front of the webcam -- no mouse, no controller. "
            "This walkthrough covers the controls for each game before you "
            "step up to a station."
        ),
        controls=[
            "Stand where your hand is clearly visible to the camera",
            "Keep your hand inside the center of the frame for best tracking",
            "Each game shows an on-screen cursor or lens that follows your hand",
        ],
        shortcuts=[
            "Esc closes the current game and returns to the arcade hub",
        ],
        accent=PLEXUS_GRAY,
    ),
    TutorialPage(
        game_title="SMT Pick and Place",
        subtitle="Circuit Builder",
        objective=(
            "Place each component into its matching socket on the board, "
            "just like a real SMT pick-and-place line -- but with your fingers "
            "instead of a robotic arm."
        ),
        controls=[
            "Bring your thumb and index finger together to PINCH and grab a component",
            "Move your hand to drag the grabbed component toward its socket",
            "Spread your thumb and index finger apart to RELEASE and drop it in place",
            "A white ring cursor means you're not pinching; it turns green and fills solid when you are",
        ],
        shortcuts=[
            "R resets the board",
            "D toggles the debug HUD",
            "P toggles the webcam picture-in-picture feed",
            "Esc quits to the hub",
        ],
        accent=PLEXUS_RED,
    ),
    TutorialPage(
        game_title="SMT Reflow",
        subtitle="Factory Navigator",
        objective=(
            "Steer a PCB down the conveyor through the reflow oven, dodging "
            "hazards along the way. Reach 100 meters with your board intact "
            "to complete the run."
        ),
        controls=[
            "Stand back so your full upper body is visible to the camera",
            "Lean your shoulders LEFT to steer the board left",
            "Lean your shoulders RIGHT to steer the board right",
            "Stay centered and level to hold your lane",
            "Colliding with a hazard costs one of your 3 HP and briefly slows the belt",
            "Reach the finish line before your HP runs out to win",
        ],
        shortcuts=[
            "Space or Enter starts the run from the title screen",
            "R restarts after a completed or failed run",
            "D toggles the telemetry debug HUD",
            "Esc quits to the hub",
        ],
        accent="#F39C12",
    ),
    TutorialPage(
        game_title="AOI Inspection",
        subtitle="AOI Inspector",
        objective=(
            "A circuit board appears on screen. Decide whether it passes "
            "quality inspection or should be rejected -- just like an "
            "automated optical inspection station."
        ),
        controls=[
            "Show a THUMBS UP to a board to PASS it",
            "Show a THUMBS DOWN to REJECT it",
            "Hold the gesture steady -- a confidence bar fills up as it locks in",
            "Once the bar fills completely, your decision is locked in automatically",
        ],
        shortcuts=[
            "Space or Enter starts the round from the title screen",
            "R restarts after a completed run",
            "D toggles the debug HUD (shows FPS and detected gesture)",
            "Esc quits to the hub",
        ],
        accent=PLEXUS_GREEN,
    ),
    TutorialPage(
        game_title="X-Ray Inspection",
        subtitle="Defect Detective",
        objective=(
            "Sweep a handheld X-ray lens across the board to reveal hidden "
            "defects, then flag every one you find."
        ),
        controls=[
            "Move your hand to steer the circular X-ray lens over the board",
            "The lens reveals the X-ray view only in the area directly under it",
            "Pinch your thumb and index finger together to FLAG a defect under the lens",
            "Found defects are outlined in green and marked FOUND",
            "Find every defect on the board to win",
        ],
        shortcuts=[
            "R resets the board",
            "D toggles the debug HUD",
            "P toggles an on-screen panel",
            "Esc quits to the hub",
        ],
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

        self._configure_window()
        self.body = ft.Column(expand=True)
        self.page.add(self.body)
        self._render()

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
            width=900,
            padding=50,
            border_radius=24,
            bgcolor=ft.Colors.with_opacity(0.55, "#111111"),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.15, PLEXUS_WHITE)),
            content=ft.Column(
                spacing=22,
                controls=[
                    ft.Row(
                        controls=[
                            ft.Container(
                                width=6, height=36, border_radius=3, bgcolor=p.accent
                            ),
                            ft.Column(
                                spacing=0,
                                controls=[
                                    ft.Text(
                                        p.subtitle.upper(),
                                        style=ft.TextStyle(
                                            size=13,
                                            weight=ft.FontWeight.BOLD,
                                            color=p.accent,
                                            letter_spacing=2,
                                        ),
                                    ),
                                    ft.Text(
                                        p.game_title,
                                        size=28,
                                        weight=ft.FontWeight.W_900,
                                        color=PLEXUS_WHITE,
                                    ),
                                ],
                            ),
                        ]
                    ),
                    ft.Container(
                        padding=ft.Padding.symmetric(vertical=4),
                        content=ft.Text(
                            p.objective,
                            size=15,
                            color=ft.Colors.WHITE70,
                            italic=p.is_placeholder,
                        ),
                    ),
                    ft.Divider(color=ft.Colors.with_opacity(0.1, PLEXUS_WHITE)),
                    ft.Text(
                        "HOW TO PLAY", 
                        style=ft.TextStyle(
                            size=12, 
                            weight=ft.FontWeight.BOLD, 
                            color=ft.Colors.WHITE54, 
                            letter_spacing=2
                        )
                    ),
                    self._bullet_list(p.controls, ft.Icons.PAN_TOOL_ALT, p.accent),
                    ft.Text(
                        "SHORTCUTS", 
                        style=ft.TextStyle(
                            size=12, 
                            weight=ft.FontWeight.BOLD, 
                            color=ft.Colors.WHITE54, 
                            letter_spacing=2
                        )
                    ),
                    self._bullet_list(p.shortcuts, ft.Icons.KEYBOARD, ft.Colors.WHITE54),
                ],
            ),
        )

        nav = ft.Row(
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=18,
            controls=[
                ft.OutlinedButton(
                    "Back",
                    on_click=lambda e: self._prev(),
                    disabled=self.index == 0,
                    style=ft.ButtonStyle(color=PLEXUS_WHITE),
                ),
                ft.Row(spacing=8, controls=[self._dot(i) for i in range(len(PAGES))]),
                ft.FilledButton(
                    "Start Playing" if is_last else "Next",
                    on_click=lambda e: self._next(),
                    style=ft.ButtonStyle(
                        bgcolor=PLEXUS_RED if is_last else PLEXUS_GRAY,
                        color=PLEXUS_WHITE,
                    ),
                ),
            ],
        )

        self.body.controls = [
            ft.Container(
                expand=True,
                alignment=ft.Alignment(0, 0),
                content=ft.Column(
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=32,
                    controls=[card, nav],
                ),
            )
        ]
        self.page.update()


def main(page: ft.Page) -> None:
    TutorialApp(page)


if __name__ == "__main__":
    ft.run(main)
