#!/usr/bin/env python3
"""
gesture_fusion.py

Combined overlay + gesture fusion + in-window AirKeyboard toggle.

Requirements:
- Do NOT modify hand_tracking_backend.py
- This module starts camera (via the backend), uses the backend's capture_thread,
  then runs a single main display loop that:
    - Draws gesture HUD + hotspots + hints (re-uses backend logic where possible)
    - Provides a bottom-right Toggle Keyboard button
    - Draws an overlay keyboard (when toggled) and accepts pinch taps to type

Notes:
- All OpenCV imshow calls happen in this file only (avoids multi-thread imshow issues).
- The backend still exposes helpers and state which we import and reuse.
"""

import time
import threading
import math
import platform
import cv2
from PyQt5.QtWidgets import QApplication
from ai_keyboard import AirKeyboard
# Import backend functions/globals — we will not call its processing_loop to avoid double imshow
import hand_tracking_backend as htb
from hand_tracking_backend import (
    load_config,
    save_config,
    calibrate_interactive,
    capture_thread,
    hotspots,
    pinch_threshold,
)

# Import AI keyboard backend
try:
    from ai_keyboard import AirKeyboard
except Exception:
    # minimal fallback if ai_keyboard missing — simple class
    class AirKeyboard:
        def __init__(self, *args, **kwargs):
            self.typed_text = ""
            self.key_map = self._default_key_map()

        def detect_hover(self, pos): return self._map_finger_to_key(pos)
        def detect_tap(self, pos, pinch): 
            k = self._map_finger_to_key(pos)
            if pinch and k:
                self.key_pressed(k)
                return k
            return None
        def key_pressed(self, k):
            self.typed_text += k
        def predict_next(self): return []
        def _map_finger_to_key(self,pos):
            for k,p in self.key_map.items():
                x,y,w,h = p
                if x <= pos[0] <= x+w and y <= pos[1] <= y+h:
                    return k
            return None
        def _default_key_map(self):
            keys="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            return {k:(i*40,0,38,38) for i,k in enumerate(keys)}

# Config
FRAME_W = htb.FRAME_WIDTH
FRAME_H = htb.FRAME_HEIGHT

# Keyboard/layout settings (overlay coords are in camera frame)
KB_ROWS = [
    "1234567890",
    "QWERTYUIOP",
    "ASDFGHJKL",
    "ZXCVBNM",
    "< SPACE >", 
    "< CLOSE >" 
]
KB_KEY_W = 48
KB_KEY_H = 46
KB_MARGIN = 8
KB_AREA_H = (KB_KEY_H + KB_MARGIN) * 3 + 60

# Toggle button (bottom-right) size in camera window coordinates
TOGGLE_W = 180
TOGGLE_H = 54
TOGGLE_MARGIN = 14  # margin from bottom-right

# Shared state
keyboard_open = False
typed_display = ""  # text shown on overlay
last_typed_time = 0.0

# small helpers
def clamp(v,a,b): return max(a, min(b, v))
def pixel_distance(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

# Map normalized webcam coords to screen using backend helper (but our keyboard uses camera overlay coords)
def normalized_to_cam(nx, ny, cam_w, cam_h):
    tx = clamp(nx, 0.0, 1.0) * cam_w
    ty = clamp(ny, 0.0, 1.0) * cam_h
    return int(tx), int(ty)

# Draw toggle button on frame bottom-right
def draw_toggle_button(frame, text="Toggle Keyboard", hovered=False):
    fh, fw = frame.shape[:2]
    x2 = fw - TOGGLE_MARGIN
    y2 = fh - TOGGLE_MARGIN
    x1 = x2 - TOGGLE_W
    y1 = y2 - TOGGLE_H
    color = (80, 200, 120) if hovered else (50, 120, 200)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x1+12, y1+TOGGLE_H//2+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return (x1,y1,x2,y2)

# Draw keyboard overlay at bottom area in camera coords
def draw_keyboard(frame, hover_key=None):
    fh, fw = frame.shape[:2]
    kb_w = fw - 2*20
    kb_x = 20
    kb_y = fh - KB_AREA_H - 10
    # background
    cv2.rectangle(frame, (kb_x-6, kb_y-6), (kb_x + kb_w + 6, kb_y + KB_AREA_H + 6), (20,20,20, ), -1)
    # draw rows
    start_y = kb_y + 18
    key_positions = {}  # key -> (x,y,w,h)
    for r_idx, row in enumerate(KB_ROWS):
        row_len = len(row)
        # center the row
        total_w = row_len * KB_KEY_W + (row_len-1) * KB_MARGIN
        start_x = kb_x + (kb_w - total_w)//2
        y = start_y + r_idx * (KB_KEY_H + KB_MARGIN)
        for i, ch in enumerate(row.split()):
            if ch == "<SPACE>":
                kw = KB_KEY_W * 5 + KB_MARGIN * 4
                label = "SPACE"
            elif ch == "<CLOSE>":
                kw = KB_KEY_W * 2
                label = "CLOSE"
            else:
                kw = KB_KEY_W
                label = ch

            x = start_x + i * (kw + KB_MARGIN)
            bg = (140,140,140) if hover_key == ch else (230,230,230)
            cv2.rectangle(frame, (x, y), (x + kw, y + KB_KEY_H), bg, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + kw, y + KB_KEY_H), (60, 60, 60), 1, cv2.LINE_AA)
            cv2.putText(frame, label, (x + 10, y + KB_KEY_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
            key_positions[ch] = (x, y, kw, KB_KEY_H)

    # typed text area
    tt_x = kb_x + 14
    tt_y = kb_y + KB_AREA_H - 30
    cv2.rectangle(frame, (tt_x, tt_y-28), (tt_x + kb_w - 28, tt_y+4), (10,10,10), -1)
    cv2.putText(frame, air_kb.typed_text[-48:], (tt_x+6, tt_y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
    return key_positions, (kb_x, kb_y, kb_w, KB_AREA_H)

# Main combined loop (single imshow owner)
def main_loop():
    global keyboard_open, air_kb, typed_display, last_typed_time

    # timing for FPS
    last_t = time.time()
    fps = 0
    app = QApplication([])
    air_kb = AirKeyboard()
    air_kb.show_keyboard()

    # some local state copied from backend defaults
    cfg = load_config()
    cal_data = cfg.get("calibration_data")
    if not cal_data:
        # fallback calibrations if backend didn't provide (should have)
        cal_data = {"Top-Left": (0.05,0.12), "Bottom-Right": (0.95,0.88)}
    cal_tl = cal_data.get("Top-Left")
    cal_br = cal_data.get("Bottom-Right")

    # main loop
    while not htb.shutdown_flag.is_set():
        with htb.frame_lock:
            frame = htb._latest_frame.copy() if htb._latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        fh, fw = frame.shape[:2]
        display = frame.copy()
        now = time.time()

        # compute FPS (simple)
        dt = now - last_t if (now - last_t) > 1e-9 else 1e-9
        last_t = now
        fps = int(0.9 * fps + 0.1 * (1.0 / dt))

        # process hands using backend's MediaPipe instance
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        results = htb.hands.process(rgb)

        # variables for UI / gestures
        visual_hint = ""
        gesture_hints = []
        hover_toggle = False
        toggle_rect = None
        hovered_key = None
        index_cam_pos = None
        pinch_active = False

        if results and results.multi_hand_landmarks:
            # iterate first hand to find index/thumb etc. We will still show all hands for landmarks
            for hand_idx, lm in enumerate(results.multi_hand_landmarks):
                htb.mp_draw.draw_landmarks(display, lm, htb.mp.solutions.hands.HAND_CONNECTIONS)
            # main first-hand logic (use 0)
            lm = results.multi_hand_landmarks[0]
            thumb = lm.landmark[4]
            index = lm.landmark[8]
            middle = lm.landmark[12]
            ring = lm.landmark[16]
            pinky = lm.landmark[20]

            ix_px, iy_px = int(index.x * fw), int(index.y * fh)
            tx_px, ty_px = int(thumb.x * fw), int(thumb.y * fh)
            index_cam_pos = (ix_px, iy_px)

            # distances
            dist_thumb_index = pixel_distance((tx_px,ty_px),(ix_px,iy_px))
            dist_thumb_pinky = pixel_distance((tx_px,ty_px),(int(pinky.x*fw), int(pinky.y*fh)))

            # determine up fingers roughly (relative y of landmarks)
            is_index_up = index.y < lm.landmark[5].y
            is_middle_up = middle.y < lm.landmark[9].y
            is_ring_up = ring.y < lm.landmark[13].y
            is_pinky_up = pinky.y < lm.landmark[17].y

            # pinch detection using backend pinch_threshold variable
            if dist_thumb_index < htb.pinch_threshold:
                pinch_active = True

            # common visual markers
            cv2.circle(display, (ix_px, iy_px), 8, (255,0,255), -1)

            # reuse palm swipe detection from backend by reading stored prev centers
            # compute hand_center_x as backend
            hand_center_x = (lm.landmark[0].x + lm.landmark[9].x)/2
            prev_x_center = htb.prev_hand_centers.get(0, hand_center_x)
            dx = hand_center_x - prev_x_center
            htb.prev_hand_centers[0] = hand_center_x
            if abs(dx) > 0.15 and (time.time() - htb.last_palm_gesture_time) > htb.PALM_GESTURE_COOLDOWN:
                direction = "PALM_LEFT" if dx < 0 else "PALM_RIGHT"
                gesture_hints.append(direction)
                htb.perform_gesture_action(direction)
                htb.last_palm_gesture_time = time.time()

            # right click gesture (three-finger pinch-like)
            if is_index_up and is_middle_up and is_ring_up and dist_thumb_index < htb.pinch_threshold*1.15:
                if (time.time() - htb.last_right_click_time) > htb.RIGHT_CLICK_COOLDOWN:
                    htb.mouse.click(htb.Button.right, 1)
                    htb.last_right_click_time = time.time()
                    visual_hint = "Right Click"

            # scroll gesture (two-finger)
            if is_index_up and is_middle_up and not is_ring_up and not is_pinky_up:
                curr_scroll = (index.y * fh, index.x * fw)
                if htb.last_scroll_pos is not None:
                    dy = htb.last_scroll_pos[0] - curr_scroll[0]
                    dx = htb.last_scroll_pos[1] - curr_scroll[1]
                    if abs(dy) > abs(dx):
                        amount = int(dy * htb.SCROLL_SENSITIVITY)
                        if amount != 0:
                            htb.mouse.scroll(0, amount)
                            visual_hint = "Scroll V"
                    else:
                        amount = int(dx * htb.SCROLL_SENSITIVITY)
                        if amount != 0:
                            htb.mouse.scroll(amount, 0)
                            visual_hint = "Scroll H"
                htb.last_scroll_pos = curr_scroll
            else:
                htb.last_scroll_pos = None

            # zoom (thumb-pinky)
            if dist_thumb_pinky > 8:
                if not htb.zoom_active:
                    htb.zoom_active = True
                    htb.zoom_base = dist_thumb_pinky
                else:
                    delta = dist_thumb_pinky - htb.zoom_base
                    if abs(delta) > 4:
                        smooth_delta = delta * htb.HZOOM_SENSITIVITY
                        htb.keyboard.press(htb.ZOOM_MOD_KEY)
                        htb.mouse.scroll(0, round(smooth_delta))
                        htb.keyboard.release(htb.ZOOM_MOD_KEY)
                        htb.zoom_base += smooth_delta * 0.6
                        htb.current_zoom_percent = clamp(int((htb.zoom_base / 100.0) * 100 + 100), 10, 500)
                        visual_hint = f"Zoom {htb.current_zoom_percent}%"
            else:
                htb.zoom_active = False

            # L-gesture lock
            if is_index_up and (thumb.y < lm.landmark[2].y) and not is_middle_up and not is_ring_up and not is_pinky_up:
                visual_hint = "L → Lock"
                htb.perform_gesture_action("L_LOCK")
                # (perform_gesture_action already handles platform locking if configured)

            # pinch click/drag
            if pinch_active:
                if not htb.is_pinching:
                    htb.is_pinching = True
                    htb.pinch_start_time = time.time()
                    visual_hint = "Pinch start"
                else:
                    held = time.time() - htb.pinch_start_time
                    if held >= htb.DRAG_HOLD_DURATION and not htb.is_dragging:
                        htb.mouse.press(htb.Button.left)
                        htb.is_dragging = True
                        visual_hint = "Drag"
            else:
                if htb.is_pinching:
                    if htb.is_dragging:
                        htb.mouse.release(htb.Button.left)
                        htb.is_dragging = False
                        visual_hint = "Drag End"
                    else:
                        if time.time() - htb.last_click_time <= htb.DOUBLE_CLICK_INTERVAL:
                            htb.mouse.click(htb.Button.left, 2)
                            visual_hint = "Double Click"
                        else:
                            htb.mouse.click(htb.Button.left, 1)
                            visual_hint = "Click"
                        htb.last_click_time = time.time()
                htb.is_pinching = False

            # cursor mapping: using same calibration mapping as backend but map to camera for hover/button detection
            cal_tl_x, cal_tl_y = cal_tl
            cal_br_x, cal_br_y = cal_br
            denom_x = (cal_br_x - cal_tl_x) if abs(cal_br_x - cal_tl_x) > 1e-6 else 1e-6
            denom_y = (cal_br_y - cal_tl_y) if abs(cal_br_y - cal_tl_y) > 1e-6 else 1e-6
            normalized_x = (index.x - cal_tl_x) / denom_x
            normalized_y = (index.y - cal_tl_y) / denom_y
            normalized_x = clamp(normalized_x, 0.0, 1.0)
            normalized_y = clamp(normalized_y, 0.0, 1.0)
            # map to camera overlay coords
            cam_x, cam_y = normalized_to_cam(normalized_x, normalized_y, fw, fh)
            # show small cursor indicator on overlay
            cv2.circle(display, (cam_x, cam_y), 6, (0,255,0), -1)

            # Check toggle button hover / pinch toggle
            tr = draw_toggle_button  # just to calc rect (draw later) - but we need coordinates:
            x1 = fw - TOGGLE_MARGIN - TOGGLE_W
            y1 = fh - TOGGLE_MARGIN - TOGGLE_H
            x2 = x1 + TOGGLE_W
            y2 = y1 + TOGGLE_H
            if x1 <= cam_x <= x2 and y1 <= cam_y <= y2:
                hover_toggle = True
                # if pinch while hovering -> toggle keyboard
                if pinch_active and (time.time() - htb.last_palm_gesture_time) > 0.3:
                    keyboard_open = not keyboard_open
                    # small debounce
                    htb.last_palm_gesture_time = time.time()
            # if keyboard is open, draw keys and handle hover/tap on keys
            if keyboard_open:
                key_positions, kb_area = draw_keyboard(display, None)
                # find hovered key by camera coords
                for k, (kx,ky,kw,kh) in key_positions.items():
                    if kx <= cam_x <= kx+kw and ky <= cam_y <= ky+kh:
                        hovered_key = k
                        break
                        # if pinch -> type it
                        if pinch_active and (time.time() - last_typed_time) > 0.25:
                            if hovered_key == "<SPACE>":
                                air_kb.key_pressed(" ")
                            elif hovered_key == "<CLOSE>":
                                keyboard_open = False
                            else:
                                air_kb.key_pressed(hovered_key)
                            typed_display = air_kb.typed_text[-48:]

                            preds = air_kb.predict_next()
                            if preds:
                                cv2.putText(display, " | ".join(preds[:3]), (50, fh - 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)

                            last_typed_time = time.time()
                            visual_hint = f"Typed: {hovered_key}"
                        break
            else:
                # draw nothing keyboard-related
                pass

        else:
            # no hand — show hint if needed
            pass

        # Draw HUD (FPS / Zoom / Hotspots / Visual hint)
        cv2.putText(display, f"FPS: {fps}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,0), 2, cv2.LINE_AA)
        cv2.putText(display, f"Zoom: {int(htb.current_zoom_percent)}%", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2, cv2.LINE_AA)
        if visual_hint:
            cv2.putText(display, visual_hint, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,200,255), 2, cv2.LINE_AA)
        # draw hotspots
        for (hx,hy) in hotspots:
            cam_x = int((hx / (htb.screen_w or 1920)) * fw)
            cam_y = int((hy / (htb.screen_h or 1080)) * fh)
            cv2.circle(display, (cam_x, cam_y), 8, (0,165,255), 1)

        # Draw toggle button now (with hover state)
        toggle_rect = draw_toggle_button(display, text="Toggle Keyboard", hovered=hover_toggle)

        # If keyboard open, re-draw keyboard with hovered key (so hover highlighted)
        if keyboard_open:
            key_positions, kb_area = draw_keyboard(display, hover_key=hovered_key)

        # show currently active gesture hints in bottom-left
        if gesture_hints:
            txt = " | ".join(gesture_hints)
            cv2.putText(display, txt, (10, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2, cv2.LINE_AA)

        # Render final overlay
        cv2.imshow("Hand-Control (fusion)", display)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            # exit
            htb.shutdown_flag.set()
            break
        elif k == ord('r'):
            # recalibrate
            new_cal, new_thresh = calibrate_interactive()
            if new_cal:
                cfg = load_config()
                cfg["calibration_data"] = new_cal
                cfg["pinch_threshold"] = new_thresh
                cfg.setdefault("hotspots", []).extend(hotspots)
                save_config(cfg)

    # final cleanup
    cv2.destroyAllWindows()


def detect_camera_backend():
    """
    Cross-platform camera detection. Tries several OpenCV backends and indices.
    Returns (index, backend_flag) or (None, None)
    """
    system = platform.system().lower()
    backend_choices = {
        "darwin": [cv2.CAP_AVFOUNDATION, cv2.CAP_QT, cv2.CAP_ANY],
        "linux": [cv2.CAP_V4L2, cv2.CAP_ANY],
        "windows": [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY],
    }.get(system, [cv2.CAP_ANY])
    for b in backend_choices:
        for idx in range(0,4):
            cap_test = cv2.VideoCapture(idx, b)
            if cap_test.isOpened():
                cap_test.release()
                return idx, b
    return None, None


def main():
    # set screen size in backend
    try:
        sw, sh = htb.pyautogui_size()
    except Exception:
        sw, sh = 1920, 1080
    htb.screen_w = sw
    htb.screen_h = sh

    # detect camera
    idx, backend = detect_camera_backend()
    if idx is None:
        print("ERROR: no camera detected")
        return
    # assign cap in backend module
    htb.cap = cv2.VideoCapture(idx, backend)
    htb.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    htb.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not htb.cap.isOpened():
        print("ERROR: camera open failed")
        return

    # create windows here (main loop will imshow one window)
    try:
        cv2.startWindowThread()
        cv2.namedWindow("Hand-Control (fusion)", cv2.WINDOW_NORMAL)
    except Exception:
        pass

    # load config + calibrate if needed (we reuse backend routines)
    cfg = load_config()
    cal_data = cfg.get("calibration_data")
    if not cal_data:
        cal_data, computed = calibrate_interactive()
        if cal_data:
            cfg = cfg or {}
            cfg["calibration_data"] = cal_data
            cfg["pinch_threshold"] = computed
            cfg.setdefault("hotspots", []).extend(hotspots)
            save_config(cfg)
        else:
            print("Calibration aborted.")
            return

    # start backend capture thread (fills htb._latest_frame)
    tcap = threading.Thread(target=capture_thread, daemon=True)
    tcap.start()

    # run main combined display loop (owns imshow)
    try:
        main_loop()
    except KeyboardInterrupt:
        htb.shutdown_flag.set()
    finally:
        # release and save config
        time.sleep(0.05)
        if htb.cap:
            htb.cap.release()
        cfg = load_config()
        cfg["hotspots"] = hotspots
        cfg["pinch_threshold"] = pinch_threshold
        save_config(cfg)
        print("Shutdown complete.")


if __name__ == "__main__":
    main()