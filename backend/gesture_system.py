#!/usr/bin/env python3
"""
gesture_system.py
- Single-process, single-GUI event loop integration of MediaPipe/OpenCV + PyQt5 AirKeyboard.
- Camera feed rendered inside a QLabel; gesture detection runs on each frame.
- Toggle keyboard via floating button OR raising Index + Little finger.
- Keeps many gesture features from your backend (pinch click/drag, three-finger right click,
  two-finger scroll, thumb-pinky zoom, palm swipe for app switch, L-gesture lock).
"""

import sys
import time
import math
import platform
from pathlib import Path

# Qt / UI
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap

# OpenCV / MediaPipe
import cv2
import mediapipe as mp
import numpy as np

# Input control
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

# AirKeyboard UI from your ai_keyboard.py
from ai_keyboard import AirKeyboard

# ---- Parameters ----
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TIMER_MS = 30  # ~33 FPS
PINCH_PIXEL_THRESH = 30
SCROLL_SENSITIVITY = 1.8
HZOOM_SENSITIVITY = 0.02
DOUBLE_CLICK_INTERVAL = 0.40
RIGHT_CLICK_COOLDOWN = 0.5
PALM_SWIPE_COOLDOWN = 0.7

# platform modifier for zoom
_system = platform.system().lower()
if "darwin" in _system:
    ZOOM_MOD_KEY = Key.cmd
else:
    ZOOM_MOD_KEY = Key.ctrl

# ---- Global helpers ----
mouse = MouseController()
keyboard = KeyboardController()

def clamp(v, a, b): return max(a, min(b, v))
def pixel_dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

# ---- GestureAction mapping (you can edit gestures.json instead if you want later) ----
def perform_action(action_name):
    try:
        sys_pl = platform.system().lower()
        if action_name == "switch_next":
            if sys_pl.startswith("win"):
                keyboard.press(Key.alt); keyboard.press(Key.tab)
                keyboard.release(Key.tab); keyboard.release(Key.alt)
            elif sys_pl.startswith("darwin"):
                # macOS app switch (Cmd+Tab)
                import subprocess
                subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 48 using {command down}'])
            else:
                import os; os.system('xdotool key alt+Tab')
        elif action_name == "switch_prev":
            if sys_pl.startswith("win"):
                keyboard.press(Key.alt); keyboard.press(Key.shift); keyboard.press(Key.tab)
                keyboard.release(Key.tab); keyboard.release(Key.shift); keyboard.release(Key.alt)
            elif sys_pl.startswith("darwin"):
                import subprocess
                subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 48 using {shift down, command down}'])
            else:
                import os; os.system('xdotool key alt+Shift+Tab')
        elif action_name == "lock_screen":
            if sys_pl.startswith("win"):
                import os; os.system("rundll32.exe user32.dll,LockWorkStation")
            elif sys_pl.startswith("darwin"):
                import os; os.system("pmset displaysleepnow")
            else:
                import os; os.system("gnome-screensaver-command -l")
        elif action_name == "left_click":
            mouse.click(Button.left, 1)
        elif action_name == "right_click":
            mouse.click(Button.right, 1)
        elif action_name == "scroll":
            mouse.scroll(0, 5)
        elif action_name == "zoom_in":
            keyboard.press(ZOOM_MOD_KEY); keyboard.press('+'); keyboard.release('+'); keyboard.release(ZOOM_MOD_KEY)
    except Exception as e:
        print("perform_action error:", e)

# ---- Main Window with Video Feed ----
class CameraWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        self.setStyleSheet("background: black;")
        self.setAlignment(Qt.AlignCenter)

class GestureSystemWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture System — Unified")
        self.setGeometry(60, 60, FRAME_WIDTH + 20, FRAME_HEIGHT + 120)
        central = QWidget()
        v = QVBoxLayout(central)
        self.cam_label = CameraWidget()
        v.addWidget(self.cam_label, alignment=Qt.AlignCenter)

        # Floating toggle button (keeps on top)
        self.toggle_btn = QPushButton("⌨️")
        self.toggle_btn.setFixedSize(56, 56)
        self.toggle_btn.setStyleSheet("""
            QPushButton { background-color: #7375db; color: white; border-radius: 28px; font-size:18px; }
            QPushButton:hover { background-color: #acd9da; color: #100d28; }
        """)
        self.toggle_btn.clicked.connect(self.toggle_keyboard)

        # place button inside layout for simplicity
        v.addWidget(self.toggle_btn, alignment=Qt.AlignLeft)

        self.setCentralWidget(central)

        # AirKeyboard (will be shown/hidden)
        self.keyboard_ui = AirKeyboard(preload_words=True)
        self.keyboard_ui.hide_keyboard()

        # MediaPipe + capture
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.6,
                                                 min_tracking_confidence=0.6,
                                                 max_num_hands=2)
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            print("ERROR: camera not available.")
        # state for gestures
        self.prev_click_time = 0.0
        self.prev_right_click = 0.0
        self.prev_palm_center = {}
        self.last_palm_swipe_time = 0.0
        self.is_pinching = False
        self.pinch_start = 0.0
        self.is_dragging = False
        self.last_scroll_pos = None
        self.zoom_active = False
        self.zoom_base = 0.0
        self.current_zoom_percent = 100

        # timer for frame processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(TIMER_MS)

    def toggle_keyboard(self):
        if self.keyboard_ui.isVisible():
            self.keyboard_ui.hide_keyboard()
        else:
            self.keyboard_ui.show_keyboard()

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(frame_rgb)
        fh, fw = frame.shape[:2]
        gesture_hints = []
        now = time.time()

        index_tip = None
        thumb_tip = None
        pinky_tip = None
        middle_tip = None
        ring_tip = None

        if results.multi_hand_landmarks:
            # iterate hands; we'll process only first for cursor mapping, but gestures may use others
            for hid, lm in enumerate(results.multi_hand_landmarks):
                # draw
                self.mp_draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                # compute key landmarks
                tip_index = lm.landmark[8]; tip_thumb = lm.landmark[4]
                tip_middle = lm.landmark[12]; tip_ring = lm.landmark[16]; tip_pinky = lm.landmark[20]
                ix, iy = int(tip_index.x * fw), int(tip_index.y * fh)
                tx, ty = int(tip_thumb.x * fw), int(tip_thumb.y * fh)
                px, py = int(tip_pinky.x * fw), int(tip_pinky.y * fh)
                mx, my = int(tip_middle.x * fw), int(tip_middle.y * fh)
                rx, ry = int(tip_ring.x * fw), int(tip_ring.y * fh)

                # save last seen for gesture logic
                index_tip = (ix, iy); thumb_tip = (tx, ty); pinky_tip = (px, py)
                middle_tip = (mx, my); ring_tip = (rx, ry)

                # fingertip visual
                cv2.circle(frame, (ix, iy), 8, (255, 0, 255), -1)

                # palm swipe detection (hand center using landmark 0 and 9 like backend)
                center_x = (lm.landmark[0].x + lm.landmark[9].x)/2
                prev_center = self.prev_palm_center.get(hid, center_x)
                dx = center_x - prev_center
                self.prev_palm_center[hid] = center_x
                if abs(dx) > 0.12 and (now - self.last_palm_swipe_time) > PALM_SWIPE_COOLDOWN:
                    direction = "PALM_LEFT" if dx < 0 else "PALM_RIGHT"
                    # map to app switch
                    if direction == "PALM_LEFT":
                        perform_action("switch_prev")
                    else:
                        perform_action("switch_next")
                    self.last_palm_swipe_time = now
                    gesture_hints.append(direction)

            # --- Gesture rules (single-hand heuristics) ---
            # thumb-index pinch distance => left click / drag
            if index_tip and thumb_tip:
                d_thumb_index = pixel_dist(index_tip, thumb_tip)
                if d_thumb_index < PINCH_PIXEL_THRESH:
                    # pinch start
                    if not self.is_pinching:
                        self.is_pinching = True
                        self.pinch_start = now
                        gesture_hints.append("Pinch start")
                    else:
                        # hold -> drag
                        if (now - self.pinch_start) >= 0.22 and not self.is_dragging:
                            mouse.press(Button.left)
                            self.is_dragging = True
                            gesture_hints.append("Drag start")
                else:
                    # pinch released
                    if self.is_pinching:
                        if self.is_dragging:
                            mouse.release(Button.left)
                            self.is_dragging = False
                            gesture_hints.append("Drag end")
                        else:
                            # click or double-click
                            if (now - self.prev_click_time) <= DOUBLE_CLICK_INTERVAL:
                                mouse.click(Button.left, 2)
                                gesture_hints.append("Double click")
                            else:
                                mouse.click(Button.left, 1)
                                gesture_hints.append("Click")
                            self.prev_click_time = now
                    self.is_pinching = False

            # three-finger right click heuristic: index+middle+ring up & thumb near index
            # We'll use simple pixel checks: if distances and relative y satisfy approx 'up'
            if index_tip and middle_tip and ring_tip and thumb_tip:
                # detect "three fingers up" by using y-coordinates (smaller y is up)
                # rough condition:
                ups = (index_tip[1] < frame.shape[0]*0.6) and (middle_tip[1] < frame.shape[0]*0.6) and (ring_tip[1] < frame.shape[0]*0.6)
                if ups and pixel_dist(index_tip, thumb_tip) < PINCH_PIXEL_THRESH * 1.2:
                    if (now - self.prev_right_click) > RIGHT_CLICK_COOLDOWN:
                        mouse.click(Button.right, 1)
                        self.prev_right_click = now
                        gesture_hints.append("Right click")

            # two-finger scroll: index + middle up and track vertical delta
            if index_tip and middle_tip:
                # use normalized y of index & middle to decide scroll
                curr_scroll = (index_tip[1], index_tip[0])  # (y, x)
                if self.last_scroll_pos is not None:
                    dy = self.last_scroll_pos[0] - curr_scroll[0]
                    dx = self.last_scroll_pos[1] - curr_scroll[1]
                    if abs(dy) > abs(dx):
                        amount = int(dy * SCROLL_SENSITIVITY)
                        if amount != 0:
                            mouse.scroll(0, amount)
                            gesture_hints.append("Scroll V")
                    else:
                        amount = int(dx * SCROLL_SENSITIVITY)
                        if amount != 0:
                            mouse.scroll(amount, 0)
                            gesture_hints.append("Scroll H")
                self.last_scroll_pos = curr_scroll

            # zoom with thumb-pinky spread
            if thumb_tip and pinky_tip:
                d_thumb_pinky = pixel_dist(thumb_tip, pinky_tip)
                # engage zoom mode if spread beyond threshold
                if d_thumb_pinky > 80:
                    if not self.zoom_active:
                        self.zoom_active = True
                        self.zoom_base = d_thumb_pinky
                    else:
                        delta = d_thumb_pinky - self.zoom_base
                        if abs(delta) > 4:
                            smooth_delta = delta * HZOOM_SENSITIVITY
                            keyboard.press(ZOOM_MOD_KEY)
                            mouse.scroll(0, round(smooth_delta))
                            keyboard.release(ZOOM_MOD_KEY)
                            # update base gradually
                            self.zoom_base += smooth_delta * 0.6
                            gesture_hints.append("Zoom")
                else:
                    self.zoom_active = False

            # L-Shape (index + thumb up, others down) -> lock screen
            # We check index tip higher than its pip and thumb above thumb_ip approx
            # heuristics only: index.y < index_pip.y and thumb.y < some landmark (we lack pip here: use landmark 2)
            # We'll approximate by comparing index tip vs landmark 6 (index pip) if available
            # (MediaPipe: index pip is landmark 6, thumb IP is 3)
            # We'll fetch those if possible:
            # NOTE: above we only stored tips. To be precise, we could re-extract landmarks; using a quick pass:
            lm0 = results.multi_hand_landmarks[0]
            idx_tip_y = lm0.landmark[8].y if lm0 else None
            idx_pip_y = lm0.landmark[6].y if lm0 else None
            thumb_ip_y = lm0.landmark[3].y if lm0 else None
            mid_y = lm0.landmark[12].y if lm0 else None
            ring_y = lm0.landmark[16].y if lm0 else None
            pinky_y = lm0.landmark[20].y if lm0 else None
            if idx_tip_y and idx_pip_y and thumb_ip_y:
                if idx_tip_y < idx_pip_y and thumb_ip_y < lm0.landmark[2].y and (mid_y > idx_pip_y and ring_y > idx_pip_y and pinky_y > idx_pip_y):
                    perform_action("lock_screen")
                    gesture_hints.append("L -> lock")
                    # small cooldown
                    time.sleep(0.4)

            # Index + little finger up -> toggle keyboard (gesture-level)
            # Use tip vs pip heuristic: tip y < pip y => finger up
            try:
                # index tip 8, index pip 6, pinky tip 20, pinky pip 18
                mh = results.multi_hand_landmarks[0]
                idx_up = mh.landmark[8].y < mh.landmark[6].y
                pinky_up = mh.landmark[20].y < mh.landmark[18].y
                if idx_up and pinky_up:
                    # toggle keyboard (user requested this)
                    # to avoid flipping every frame, add small cooldown
                    if not hasattr(self, "_kbd_toggle_last") or (now - self._kbd_toggle_last) > 1.0:
                        self.toggle_keyboard()
                        self._kbd_toggle_last = now
                        gesture_hints.append("Index+Pinky -> toggle keyboard")
            except Exception:
                pass

        else:
            # no hands: reset some state
            self.last_scroll_pos = None
            self.zoom_active = False

        # render gesture hints overlay onto frame
        if gesture_hints:
            hint = " | ".join(gesture_hints)
            cv2.putText(frame, hint, (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)

        # convert BGR -> QImage -> pixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.cam_label.width(), self.cam_label.height(), Qt.KeepAspectRatio)
        self.cam_label.setPixmap(pix)

    def closeEvent(self, ev):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        # close keyboard UI
        try:
            self.keyboard_ui.close()
        except Exception:
            pass
        super().closeEvent(ev)

# ---- main ----
def main():
    # ensure QApplication in main thread (macOS requirement)
    app = QApplication(sys.argv)
    win = GestureSystemWindow()
    win.show()
    print("Unified gesture system started. Toggle keyboard with button or index+pinky gesture.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()