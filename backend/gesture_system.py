#!/usr/bin/env python3
"""
gesture_system.py
- Launches hand_tracking_backend.py in a separate process (backend keeps its main thread).
- Runs the AirKeyboard Qt UI in THIS process (main thread).
- Watches gesture_events.json (written by backend) and dispatches mapped actions.
- Adds a HUD window and gesture priority resolution.
"""

import os
import sys
import json
import time
import platform
import subprocess
import threading
from pathlib import Path
from collections import defaultdict

# UI / keyboard imports (must run in main thread)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QTimer

# local keyboard UI
from ai_keyboard import AirKeyboard

# input control for actions
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

ROOT = Path(__file__).resolve().parent
HAND_BACKEND = ROOT / "hand_tracking_backend.py"
GESTURE_EVENTS = ROOT / "gesture_events.json"
GESTURE_CONFIG = ROOT / "gestures.json"
BACKEND_PY = sys.executable

mouse = MouseController()
keyboard = KeyboardController()

# ---- Priority map: higher number = higher priority ----
# You can edit or extend this on the fly via gestures.json (see load_gesture_map)
DEFAULT_PRIORITY = {
    # high-level explicit gestures
    "three_finger_pinch": 95,
    "pinch": 90,
    "tap": 60,
    "hover": 40,
    "swipe_left": 70,
    "swipe_right": 70,
    "lock_gesture": 100,
    # special toggle
    "index_pinky": 85,
    # palm swipes
    "PALM_LEFT": 75,
    "PALM_RIGHT": 75,
}

# Default gesture -> action fallback map if gestures.json missing
BUILTIN_ACTIONS = {
    "swipe_left": "switch_prev",
    "swipe_right": "switch_next",
    "pinch": "left_click",
    "three_finger_pinch": "right_click",
    "hover": "scroll",
    "tap": "zoom",
    "lock_gesture": "lock_screen",
    # custom mapping for index+pinky to toggle keyboard
    "index_pinky": "keyboard_toggle",
    # palm labels (from backend processing, keep for compatibility)
    "PALM_LEFT": "switch_prev",
    "PALM_RIGHT": "switch_next",
}

# helper: read gestures.json mapping and optional priority overrides
def load_gesture_map_and_priority():
    gmap = {}
    priority = DEFAULT_PRIORITY.copy()
    if GESTURE_CONFIG.exists():
        try:
            raw = json.loads(GESTURE_CONFIG.read_text(encoding="utf-8"))
            for g in raw.get("gestures", []):
                name = g.get("name")
                action = g.get("action")
                pr = g.get("priority")
                if name and action:
                    gmap[name] = action
                if name and isinstance(pr, (int, float)):
                    priority[name] = int(pr)
        except Exception:
            pass
    # merge builtin actions for any missing mapping keys
    for k, v in BUILTIN_ACTIONS.items():
        gmap.setdefault(k, v)
    return gmap, priority

# perform OS-level actions (switch, lock, clicks, scroll, zoom)
def perform_action(action_name):
    sys_platform = platform.system().lower()
    try:
        if action_name == "switch_next":
            if sys_platform.startswith("win"):
                keyboard.press(Key.alt); keyboard.press(Key.tab)
                keyboard.release(Key.tab); keyboard.release(Key.alt)
            elif sys_platform.startswith("darwin"):
                os.system('osascript -e \'tell application "System Events" to key code 48 using {command down}\'')
            else:
                os.system('xdotool key alt+Tab')

        elif action_name == "switch_prev":
            if sys_platform.startswith("win"):
                keyboard.press(Key.alt); keyboard.press(Key.shift); keyboard.press(Key.tab)
                keyboard.release(Key.tab); keyboard.release(Key.shift); keyboard.release(Key.alt)
            elif sys_platform.startswith("darwin"):
                os.system('osascript -e \'tell application "System Events" to key code 48 using {shift down, command down}\'')
            else:
                os.system('xdotool key alt+Shift+Tab')

        elif action_name == "lock_screen":
            if sys_platform.startswith("win"):
                os.system("rundll32.exe user32.dll,LockWorkStation")
            elif sys_platform.startswith("darwin"):
                os.system("pmset displaysleepnow")
            else:
                os.system("gnome-screensaver-command -l")

        elif action_name == "left_click":
            mouse.click(Button.left, 1)
        elif action_name == "right_click":
            mouse.click(Button.right, 1)
        elif action_name == "scroll":
            mouse.scroll(0, 5)
        elif action_name == "zoom":
            keyboard.press(Key.ctrl); keyboard.press('+'); keyboard.release('+'); keyboard.release(Key.ctrl)
        else:
            # allow numeric scroll or other commands in future
            pass
    except Exception as e:
        print("perform_action error:", e)

class GestureWatcher(threading.Thread):
    """Poll gesture_events.json (written by backend) and dispatch gestures with priority."""
    def __init__(self, keyboard_ui, polling=0.12):
        super().__init__(daemon=True)
        self.keyboard_ui = keyboard_ui
        self.polling = polling
        self._stop = threading.Event()
        self._last_seen = {}  # gesture -> timestamp when we processed it
        self.gmap, self.priority = load_gesture_map_and_priority()

        # freshness TTL for events (seconds) — allows retrigger after TTL
        self._ttl = 1.2

    def stop(self):
        self._stop.set()

    def run(self):
        print("👀 Watching for gestures (gesture_events.json)...")
        while not self._stop.is_set():
            try:
                # reload mapping live in case file edited
                self.gmap, self.priority = load_gesture_map_and_priority()

                if GESTURE_EVENTS.exists():
                    raw = GESTURE_EVENTS.read_text(encoding="utf-8")
                    try:
                        events = json.loads(raw)
                    except Exception:
                        events = []
                    # support older formats where events may be dict
                    if isinstance(events, dict):
                        events = events.get("events", []) or []
                    # events might be list of names (strings)
                    if not isinstance(events, list):
                        events = [events]

                    # dedupe and pick highest-priority gestures appearing simultaneously
                    now = time.time()
                    unique = list(dict.fromkeys([str(e) for e in events if isinstance(e, (str, int))]))
                    if not unique:
                        time.sleep(self.polling)
                        continue

                    # Filter by TTL (we allow reprocessing after TTL)
                    candidates = []
                    for g in unique:
                        last_t = self._last_seen.get(g, 0.0)
                        if now - last_t > self._ttl:
                            candidates.append(g)

                    if not candidates:
                        time.sleep(self.polling)
                        continue

                    # Determine highest priority among candidates (if multiple)
                    def pr_for(gesture_name):
                        return int(self.priority.get(gesture_name, DEFAULT_PRIORITY.get(gesture_name, 10)))

                    candidates_sorted = sorted(candidates, key=lambda x: pr_for(x), reverse=True)
                    # process in priority order but allow short-circuit: if a high-prio gesture triggers toggle/show/hide,
                    # we don't want lower-priority gestures immediately after to conflict.
                    for gesture_name in candidates_sorted:
                        # mark processed timestamp immediately (prevents retrigger until TTL)
                        self._last_seen[gesture_name] = now

                        action = self.gmap.get(gesture_name)
                        if not action:
                            # fallback: some backends may emit "PALM_LEFT"/"PALM_RIGHT"
                            action = BUILTIN_ACTIONS.get(gesture_name)

                        if not action:
                            # if no known action, ignore
                            print(f"Unknown gesture (ignored): {gesture_name}")
                            continue

                        # handle keyboard actions inside UI (must call UI thread methods)
                        if action == "keyboard_toggle":
                            # toggle the keyboard in UI thread
                            print(f"Gesture -> keyboard_toggle ({gesture_name})")
                            # call UI-safe toggle via Qt event loop
                            self.keyboard_ui.toggle_from_thread()
                        elif action == "keyboard_show":
                            print(f"Gesture -> keyboard_show ({gesture_name})")
                            self.keyboard_ui.show_keyboard_from_thread()
                        elif action == "keyboard_hide":
                            print(f"Gesture -> keyboard_hide ({gesture_name})")
                            self.keyboard_ui.hide_keyboard_from_thread()
                        else:
                            # standard OS-level action
                            print(f"Gesture detected: {gesture_name} -> {action}")
                            perform_action(action)
                        # after performing a high-priority gesture, break so lower ones don't run in same cycle
                        break
                time.sleep(self.polling)
            except Exception as e:
                print("GestureWatcher error:", e)
                time.sleep(self.polling)

class HUDWindow(QWidget):
    """Single main window containing HUD status and launching/positioning the toggle button."""
    def __init__(self, keyboard_ui):
        super().__init__()
        self.keyboard_ui = keyboard_ui
        self.setWindowTitle("Gesture HUD")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setMinimumSize(360, 120)
        self.setStyleSheet("background: rgba(16,13,40,0.92); color: #acd9da; border-radius: 10px;")
        self._build_ui()
        # small timer to refresh last-gesture label
        self._last_gesture = ""
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(250)

    def _build_ui(self):
        l = QVBoxLayout()
        top = QHBoxLayout()
        self.status_label = QLabel("🎥 Backend: starting...")
        self.status_label.setStyleSheet("font-size:13px;")
        top.addWidget(self.status_label, 1)

        # small pinned toggle button inside HUD
        self.pinned_btn = QPushButton("⌨️ Toggle Keyboard")
        self.pinned_btn.setFixedHeight(34)
        self.pinned_btn.setStyleSheet("""
            QPushButton { background: #7375db; color: white; border-radius:6px; padding:6px; }
            QPushButton:hover { background: #acd9da; color: #100d28; }
        """)
        self.pinned_btn.clicked.connect(self._on_toggle)
        top.addWidget(self.pinned_btn, 0, Qt.AlignRight)

        l.addLayout(top)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #00003b;")
        l.addWidget(sep)

        self.last_label = QLabel("Last gesture: —")
        self.last_label.setStyleSheet("font-size:12px;")
        l.addWidget(self.last_label)

        self.setLayout(l)

    def set_status(self, text):
        self.status_label.setText(text)

    def set_last_gesture(self, g):
        self._last_gesture = g
        self.last_label.setText(f"Last gesture: {g}")

    def _on_toggle(self):
        if self.keyboard_ui.isVisible():
            self.keyboard_ui.hide_keyboard()
        else:
            self.keyboard_ui.show_keyboard()

    # These thread-safe wrappers call the UI in the Qt thread
    def show_keyboard_from_thread(self):
        QTimer.singleShot(0, self.keyboard_ui.show_keyboard)

    def hide_keyboard_from_thread(self):
        QTimer.singleShot(0, self.keyboard_ui.hide_keyboard)

    def toggle_from_thread(self):
        QTimer.singleShot(0, lambda: self._on_toggle())

    # refresh hook to update UI state if needed
    def _refresh(self):
        # show keyboard visibility state
        vis = "visible" if self.keyboard_ui.isVisible() else "hidden"
        self.set_status(f"Keyboard: {vis}  |  gesture_events: {GESTURE_EVENTS.name}")

def launch_backend_process():
    """Start the hand_tracking_backend.py in a separate process. Return Popen."""
    if not HAND_BACKEND.exists():
        raise FileNotFoundError(f"{HAND_BACKEND} not found")
    cmd = [BACKEND_PY, str(HAND_BACKEND)]
    # redirect stdout/stderr to file so additional terminal windows don't appear
    logf = ROOT / "backend.log"
    f = open(str(logf), "ab")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=f, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True
    )
    print(f"🎥 Launched hand-tracking backend (pid={proc.pid}), logs->{logf}")
    return proc, f

def main():
    # launch backend subprocess (it will create its own OpenCV window)
    backend_proc, log_handle = launch_backend_process()

    # Create QApplication (main thread) and AirKeyboard UI
    print("🧠 Starting Air Keyboard in main thread...")
    app = QApplication(sys.argv)
    keyboard_ui = AirKeyboard(preload_words=True)
    keyboard_ui.hide_keyboard()

    # HUD window
    hud = HUDWindow(keyboard_ui)
    hud.move(80, 80)
    hud.show()

    # small floating quick-toggle button (optional)
    # reuse keyboard_ui.toggle_from_thread via HUD toggle

    # Start gesture watcher thread
    watcher = GestureWatcher(hud)
    watcher.start()

    def cleanup():
        print("⚙️ Shutting down gesture_system...")
        watcher.stop()
        try:
            if backend_proc.poll() is None:
                backend_proc.terminate()
                try:
                    backend_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    backend_proc.kill()
        except Exception as e:
            print("Error terminating backend:", e)
        try:
            log_handle.close()
        except Exception:
            pass

    app.aboutToQuit.connect(cleanup)

    hud.set_status("Ready — backend running")
    print("⌨️ AirKeyboard + HUD ready.")
    try:
        sys.exit(app.exec_())
    finally:
        cleanup()

if __name__ == "__main__":
    main()