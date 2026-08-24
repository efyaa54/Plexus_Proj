from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import flet as ft
import psutil
import win32api
import win32con
import win32gui
import win32process

# --------------------------------------------------------------------------
# Config / constants
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("plexus_hub")

APP_TITLE = "Plexus AI Arcade"
FOCUS_TIMEOUT_S = 5.0
FOCUS_POLL_INTERVAL_S = 0.05
SHUTDOWN_GRACE_S = 0.2
CHILD_TERMINATE_TIMEOUT_S = 2.0

PLEXUS_RED = "#DA291C"
PLEXUS_CHARCOAL = "#1A1A1A"
PLEXUS_GRAY = "#333333"
PLEXUS_WHITE = "#FFFFFF"

SCORE_FILE = "player_data.json"

GAMES = [
    ("Tutorial", "tutorial.py"),
    ("SMT Pick and Place", "circuit_builder.py"),
    ("SMT Reflow", "factory_navigator.py"),
    ("AOI Inspection", "aoi_inspector.py"),
    ("X-Ray Inspection", "defect_detective.py"),
]

# --------------------------------------------------------------------------
# Data Persistence
# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# Win32 focus helpers
# --------------------------------------------------------------------------

def force_foreground(hwnd: Optional[int]) -> None:
    if not hwnd or not win32gui.IsWindow(hwnd):
        log.warning("force_foreground: invalid hwnd %r", hwnd)
        return

    foreground_hwnd = win32gui.GetForegroundWindow()
    current_thread = win32api.GetCurrentThreadId()
    foreground_thread, _ = win32process.GetWindowThreadProcessId(foreground_hwnd)

    attached = False
    try:
        if foreground_thread != current_thread:
            attached = win32process.AttachThreadInput(foreground_thread, current_thread, True)

        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

        win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
        win32gui.SetForegroundWindow(hwnd)
        win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32gui.BringWindowToTop(hwnd)
    except Exception:
        log.exception("force_foreground failed for hwnd=%s", hwnd)
    finally:
        if attached:
            win32process.AttachThreadInput(foreground_thread, current_thread, False)


def find_window_by_pid(pid: int, timeout: float = FOCUS_TIMEOUT_S) -> Optional[int]:
    found: dict[str, Optional[int]] = {"hwnd": None}

    def callback(hwnd: int, _: None) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid:
                found["hwnd"] = hwnd
                return False  
        return True

    deadline = time.time() + timeout
    while time.time() < deadline and found["hwnd"] is None:
        win32gui.EnumWindows(callback, None)
        if found["hwnd"]:
            break
        time.sleep(FOCUS_POLL_INTERVAL_S)
    return found["hwnd"]


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

# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

class ArcadeHub:
    def __init__(self, page: ft.Page):
        self.page = page
        self.game_process: Optional[subprocess.Popen] = None
        self._shutting_down = False
        
        self.player_data = load_player_data()

        self._configure_window()
        self._build_ui()

    def _configure_window(self) -> None:
        page = self.page
        page.title = APP_TITLE
        page.padding = 0
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

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True

        log.info("Shutting down...")
        if self.game_process is not None and self.game_process.poll() is None:
            self.game_process.terminate()

        self.page.window.visible = False
        self.page.update()
        time.sleep(SHUTDOWN_GRACE_S)

        kill_process_tree()
        os._exit(0)

    def show_toast(self, message: str, color: str = PLEXUS_RED) -> None:
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(message, color=PLEXUS_WHITE, weight=ft.FontWeight.BOLD),
            bgcolor=color,
            duration=2000,
        )
        self.page.snack_bar.open = True
        self.page.update()

    def start_new_session(self, e) -> None:
        name = self.name_input.value.strip()
        if not name:
            name = "Guest"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load existing data to keep leaderboard history intact
        data = load_player_data()
        data["active_player"] = {"name": name, "score": 0, "timestamp": timestamp}
        
        save_player_data(data)
        self.player_data = data
        
        self.update_score_display()
        self.show_toast(f"Session started for {name} at {timestamp}!", ft.Colors.GREEN_700)

    def update_score_display(self) -> None:
        active = self.player_data["active_player"]
        self.score_text.value = f"ACTIVE PLAYER: {active['name']}  |  SCORE: {active['score']}"
        self.page.update()

    def launch_game(self, script_name: str) -> None:
        if self.game_process is not None and self.game_process.poll() is None:
            self.show_toast("A game is already active! Close it first.")
            return

        if not Path(script_name).exists():
            self.show_toast(f"Error: {script_name} not found!")
            return

        display_name = script_name.removesuffix(".py").replace("_", " ").title()
        self.show_toast(f"Launching {display_name}...", ft.Colors.GREEN_700)

        threading.Thread(target=self._run_game, args=(script_name,), daemon=True).start()

    def _run_game(self, script_name: str) -> None:
        page = self.page
        try:
            page.on_keyboard_event = None
            self.game_process = subprocess.Popen([sys.executable, script_name])

            game_hwnd = find_window_by_pid(self.game_process.pid)
            page.window.visible = False
            page.update()

            if game_hwnd:
                force_foreground(game_hwnd)
            else:
                log.warning("Timed out waiting for %s's window", script_name)

            self.game_process.wait()
        except Exception:
            log.exception("Error running %s", script_name)
        finally:
            self.game_process = None
            page.window.visible = True
            page.window.minimized = False
            
            # --- RELOAD SCORE DATA UPON RETURN ---
            self.player_data = load_player_data()
            self.update_score_display()
            
            page.update()

            hub_hwnd = win32gui.FindWindow(None, APP_TITLE)
            force_foreground(hub_hwnd)

            page.on_keyboard_event = self._on_keyboard_event
            self.show_toast("Returned to Arcade Hub", PLEXUS_CHARCOAL)

    def _build_ui(self) -> None:

        active = self.player_data["active_player"]
        
        self.name_input = ft.TextField(
            value=self.player_data["active_player"]["name"],
            label="Player Name",
            width=250,
            border_color=PLEXUS_GRAY,
            focused_border_color=PLEXUS_RED,
            color=PLEXUS_WHITE,
            bgcolor=PLEXUS_CHARCOAL,
            height=50,
            content_padding=10
        )
        
        self.score_text = ft.Text(
            f"PLAYER: {active['name']}  |  SCORE: {active['score']}",
            color=PLEXUS_RED,
            style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD, letter_spacing=1),
        )

        def on_session_btn_hover(e: ft.HoverEvent) -> None:
            hovered = str(e.data).lower() == "true"
            e.control.bgcolor = PLEXUS_RED if hovered else PLEXUS_GRAY
            e.control.update()

        session_btn = ft.Container(
            content=ft.Text("Start New Session", weight=ft.FontWeight.BOLD, color=PLEXUS_WHITE),
            bgcolor=PLEXUS_GRAY,
            height=50,
            padding=ft.Padding.symmetric(horizontal=20),
            border_radius=8,
            alignment=ft.Alignment(0, 0),
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            on_hover=on_session_btn_hover,
            on_click=self.start_new_session,
        )

        player_registration_row = ft.Row(
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                self.name_input,
                session_btn
            ]
        )

        self.page.add(
            ft.Container(
                expand=True,
                image=ft.DecorationImage(src="hub_bg.png", fit=ft.BoxFit.COVER),
                alignment=ft.Alignment(0, 0),
                content=ft.Container(
                    padding=60,
                    width=700,
                    border_radius=100,
                    bgcolor=ft.Colors.with_opacity(0.65, PLEXUS_CHARCOAL),
                    blur=ft.Blur(5, 5, ft.BlurTileMode.MIRROR),
                    content=ft.Column(
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=15,
                        tight=True,
                        controls=[
                            ft.Image(src="plexus_logo.png", width=250, fit=ft.BoxFit.CONTAIN),
                            ft.Text(
                                "PLEXUS AI ARCADE",
                                color=PLEXUS_WHITE,
                                style=ft.TextStyle(size=36, weight=ft.FontWeight.W_900, letter_spacing=2),
                            ),
                            self.score_text,
                            ft.Container(height=10),
                            player_registration_row,
                            ft.Container(height=20),
                            *[self._plexus_button(label, script) for label, script in GAMES],
                        ],
                    ),
                ),
            )
        )

    def _plexus_button(self, text: str, script_name: str) -> ft.Container:
        def on_hover(e: ft.HoverEvent) -> None:
            hovered = e.data is True
            e.control.bgcolor = PLEXUS_RED if hovered else PLEXUS_GRAY
            e.control.scale = 1.03 if hovered else 1.0
            e.control.update()

        return ft.Container(
            content=ft.Text(text, size=16, weight=ft.FontWeight.BOLD, color=PLEXUS_WHITE),
            alignment=ft.Alignment(0, 0),
            width=400,
            height=60,
            border_radius=8,
            bgcolor=PLEXUS_GRAY,
            scale=1.0,
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            on_hover=on_hover,
            on_click=lambda e: self.launch_game(script_name),
        )

def main(page: ft.Page) -> None:
    ArcadeHub(page)

if __name__ == "__main__":
    ft.run(main)
