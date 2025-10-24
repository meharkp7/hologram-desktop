#!/usr/bin/env python3
"""
backend.py - Threaded hand-tracking backend for touchless desktop control.

Features:
- Exponential smoothing cursor control (low-latency)
- Pinch -> click (single/double), pinch-hold -> drag
- Three-finger pinch -> right-click
- Two-finger swipe -> vertical/horizontal scroll
- Thumb-pinky distance -> zoom (sends modifier+scroll for real-app zoom)
- Hotspots (snap-to)
- Calibration (4-point) + auto pinch threshold estimation
- Pause/resume with SPACE
- Visual overlay on camera
- Cross-platform modifier selection for zoom (Ctrl on Win/Linux, Cmd on macOS)
- Uses MediaPipe + OpenCV + pynput
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import json
import os
import threading
import queue
import platform
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
from utils import HandTrackingState
# ------------------------
# USER-TUNABLE PARAMETERS
# ------------------------
FRAME_WIDTH = 640       # smaller => faster
FRAME_HEIGHT = 480
CAMERA_INDEX = 0

DEFAULT_PINCH_THRESHOLD = 30            # pixels (will be adapted by calibration)
SMOOTHING_ALPHA = 0.35                  # exponential smoothing alpha (0..1) lower -> smoother but more lag
CURSOR_MOVE_THRESHOLD_PX = 2            # ignore tiny moves under this many pixels
DRAG_HOLD_DURATION = 0.22               # seconds before pinch becomes drag
DOUBLE_CLICK_INTERVAL = 0.40            # seconds for double-click detection
RIGHT_CLICK_COOLDOWN = 0.5              # seconds between right-clicks
SCROLL_SENSITIVITY = 1.8                # multiplier for scroll
HZOOM_SENSITIVITY = 0.02                # delta to multiply into zoom percent / scroll
HOTSPOT_RADIUS = 140                    # pixels to magnetize to hotspot
HOTSPOT_STRENGTH = 0.85                 # 0..1 (how strongly to snap)
HESITATION_HINT_TIME = 1.2              # seconds show hint if no input
FPS_DISPLAY = True

CONFIG_PATH = "config.json"

# ------------------------
# PLATFORM MODIFIER
# ------------------------
_system = platform.system().lower()
if "darwin" in _system:
    ZOOM_MOD_KEY = Key.cmd     # macOS uses Command for browser zoom
else:
    ZOOM_MOD_KEY = Key.ctrl    # Windows/Linux use Ctrl

# ------------------------
# GLOBAL STATE
# ------------------------
mouse = MouseController()
keyboard = KeyboardController()

cap = None
hands = None
mp_draw = None

frame_lock = threading.Lock()
_latest_frame = None
shutdown_flag = threading.Event()

# runtime variables
screen_w, screen_h = None, None
prev_x = prev_y = 0.0
last_input_time = time.time()
is_pinching = False
pinch_start_time = 0.0
last_click_time = 0.0
is_dragging = False
last_right_click_time = 0.0
last_scroll_pos = None
zoom_active = False
zoom_base = 0.0
current_zoom_percent = 100  # purely visual
hotspots = []
pinch_threshold = DEFAULT_PINCH_THRESHOLD

# ------------------------
# Utility functions
# ------------------------
def load_config(path=CONFIG_PATH):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(cfg, path=CONFIG_PATH):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)

def clamp(v, a, b): return max(a, min(b, v))

def pixel_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def normalized_to_screen(nx, ny):
    # nx,ny are normalized in webcam coordinates (0..1) - we map to screen
    tx = clamp(nx, 0.0, 1.0) * screen_w
    ty = clamp(ny, 0.0, 1.0) * screen_h
    return tx, ty

def snap_to_hotspots(x, y):
    if not hotspots:
        return x, y, False
    # choose nearest
    best = min(hotspots, key=lambda p: math.hypot(p[0]-x, p[1]-y))
    d = math.hypot(best[0]-x, best[1]-y)
    if d < HOTSPOT_RADIUS:
        nx = x + (best[0]-x) * HOTSPOT_STRENGTH
        ny = y + (best[1]-y) * HOTSPOT_STRENGTH
        return nx, ny, True
    return x, y, False

# Camera capture thread
# ------------------------
def capture_thread():
    global _latest_frame
    while not shutdown_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        frame = cv2.flip(frame, 1)
        # resize for consistent speed
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        with frame_lock:
            _latest_frame = frame
        # tiny sleep allows other thread to run
        time.sleep(0.001)
    # end capture

# ------------------------
# Calibration routine
# ------------------------
def calibrate_interactive(timeout_per_point=10.0):
    """
    Non-blocking, robust 4-point calibration. Press 'c' to capture current index
    position when you see it in the preview, or pinch to auto-capture.
    timeout_per_point: seconds to wait before skipping that point.
    Returns cal_pts dict and computed pinch threshold.
    """
    global pinch_threshold
    points = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    cal_pts = {}
    pinch_samples = []

    print("Calibration: point to each corner and either pinch or press 'c' to capture.")
    time.sleep(0.3)
    for pt in points:
        print(f" -> Place fingertip at {pt}, then pinch or press 'c' to capture.")
        start_t = time.time()
        captured = False
        stable_count = 0
        last_dist = None

        while not captured and (time.time() - start_t) < timeout_per_point:
            with frame_lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
            if frame is None:
                time.sleep(0.02)
                continue
            display = frame.copy()
            fh, fw = display.shape[:2]
            cv2.putText(display, f"Pinch or press 'c' at {pt}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            key = cv2.waitKey(1) & 0xFF

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                index = lm.landmark[8]
                thumb = lm.landmark[4]
                ix, iy = int(index.x * fw), int(index.y * fh)
                tx, ty = int(thumb.x * fw), int(thumb.y * fh)
                cv2.circle(display, (ix, iy), 8, (0,255,0), -1)

                dist = math.hypot((tx-ix),(ty-iy))
                # gather pinch samples for threshold computation if user pinches briefly
                if dist < DEFAULT_PINCH_THRESHOLD * 2:
                    pinch_samples.append(dist)

                # auto-capture when pinch close and stable
                if dist < DEFAULT_PINCH_THRESHOLD:
                    if last_dist is None or abs(dist - last_dist) < 6:
                        stable_count += 1
                    else:
                        stable_count = 0
                    last_dist = dist
                    if stable_count >= 3:  # stable for a few frames
                        cal_pts[pt] = (index.x, index.y)
                        print(f"Auto-captured {pt}: {cal_pts[pt]}")
                        time.sleep(0.3)
                        captured = True

            # manual capture (press 'c')
            if key == ord('c'):
                if results and results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    index = lm.landmark[8]
                    cal_pts[pt] = (index.x, index.y)
                    print(f"Manual-captured {pt}: {cal_pts[pt]}")
                    captured = True
                else:
                    print("No hand visible to capture; try again.")
            # allow abort
            if key == 27:
                print("Calibration aborted by user.")
                return None, None

            cv2.imshow("Calibration", display)

        if not captured:
            # fallback: use corner defaults if user timed out
            default = {
                "Top-Left": (0.05, 0.12),
                "Top-Right": (0.95, 0.12),
                "Bottom-Right": (0.95, 0.88),
                "Bottom-Left": (0.05, 0.88)
            }
            cal_pts[pt] = default.get(pt, (0.5, 0.5))
            print(f"Timed out capturing {pt}, using fallback {cal_pts[pt]}")

    cv2.destroyWindow("Calibration")

    # compute pinch threshold robustly
    if len(pinch_samples) >= 5:
        mean = np.mean(pinch_samples)
        std = np.std(pinch_samples)
        computed = max(8, int(mean + 1.1 * std))
    else:
        computed = DEFAULT_PINCH_THRESHOLD
    print(f"Calibration complete. Computed pinch threshold (px): {computed}")
    pinch_threshold = computed
    return cal_pts, computed

# ------------------------
# Processing loop (main)
# ------------------------
def processing_loop(cal_points):
    global prev_x, prev_y, last_input_time
    global is_pinching, pinch_start_time, last_click_time, is_dragging
    global last_right_click_time, last_scroll_pos, zoom_active, zoom_base, current_zoom_percent

    # map calibration normalized to use when mapping finger coords: we just use cal_tl and cal_br for scaling
    cal_tl = cal_points.get("Top-Left")
    cal_br = cal_points.get("Bottom-Right")
    if cal_tl is None or cal_br is None:
        # fallback to full-frame mapping
        cal_tl = (0.0, 0.0)
        cal_br = (1.0, 1.0)

    alpha = SMOOTHING_ALPHA
    # initialize prev position at current mouse pos
    try:
        prev_x, prev_y = mouse.position
        prev_x = float(prev_x)
        prev_y = float(prev_y)
    except Exception:
        prev_x = screen_w / 2
        prev_y = screen_h / 2

    last_visual_hint_time = 0.0
    visual_hint = ""

    while not shutdown_flag.is_set():
        with frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None
        if frame is None:
            time.sleep(0.005)
            continue

        fh, fw = frame.shape[:2]
        display = frame.copy()
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now = time.time()

        is_hand_present = False

        if results.multi_hand_landmarks:
            is_hand_present = True
            last_input_time = now

            # We only care about the first detected hand for primary controls
            lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display, lm, mp.solutions.hands.HAND_CONNECTIONS)

            # fingertip landmarks
            thumb = lm.landmark[4]
            index = lm.landmark[8]
            middle = lm.landmark[12]
            ring = lm.landmark[16]
            pinky = lm.landmark[20]

            # compute pixel positions in camera frame
            ix_px = int(index.x * fw)
            iy_px = int(index.y * fh)
            tx_px = int(thumb.x * fw)
            ty_px = int(thumb.y * fh)
            pinkx_px = int(pinky.x * fw)
            pinky_px = int(pinky.y * fh)

            # pixel distances for pinch and zoom detection
            dist_thumb_index_px = pixel_distance((tx_px, ty_px), (ix_px, iy_px))
            dist_thumb_pinky_px = pixel_distance((tx_px, ty_px), (pinkx_px, pinky_px))

            # finger 'up' heuristics (relative to finger base)
            is_index_up = index.y < lm.landmark[5].y
            is_middle_up = middle.y < lm.landmark[9].y
            is_ring_up = ring.y < lm.landmark[13].y
            is_pinky_up = pinky.y < lm.landmark[17].y

            # --- RIGHT CLICK: three-finger pinch (index+middle close and pinch present)
            if is_index_up and is_middle_up and is_ring_up and dist_thumb_index_px < pinch_threshold * 1.15:
                if (now - last_right_click_time) > RIGHT_CLICK_COOLDOWN:
                    mouse.click(Button.right, 1)
                    last_right_click_time = now
                    visual_hint = "Right Click"
                    last_visual_hint_time = now

            # --- SCROLL: two-finger mode (index+middle up) - vertical & horizontal
            if is_index_up and is_middle_up and not is_ring_up and not is_pinky_up:
                curr_scroll = (index.y * fh, index.x * fw)
                if last_scroll_pos is not None:
                    dy = last_scroll_pos[0] - curr_scroll[0]
                    dx = last_scroll_pos[1] - curr_scroll[1]
                    # choose dominant direction
                    if abs(dy) > abs(dx):
                        # vertical scroll
                        amount = int(dy * SCROLL_SENSITIVITY)
                        if amount != 0:
                            mouse.scroll(0, amount)
                            visual_hint = "Scroll V"
                            last_visual_hint_time = now
                    else:
                        amount = int(dx * SCROLL_SENSITIVITY)
                        if amount != 0:
                            mouse.scroll(amount, 0)
                            visual_hint = "Scroll H"
                            last_visual_hint_time = now
                last_scroll_pos = curr_scroll
                # draw indicator
                cv2.circle(display, (ix_px, iy_px), 14, (0, 255, 255), 2)
                # skip other actions while scrolling
                # (but still allow zoom mode detection if you want - here we prioritize scrolling)
                # continue  -> we'll not continue so other things like zoom could also act if desired
            else:
                last_scroll_pos = None

            # --- ZOOM: thumb-pinky distance based (works by sending modifier + mouse.scroll)
            # We'll convert delta of thumb-pinky into small scrolls while holding modifier (Ctrl/Cmd)
            if dist_thumb_pinky_px > 0:
                if not zoom_active:
                    zoom_active = True
                    zoom_base = dist_thumb_pinky_px
                else:
                    delta = dist_thumb_pinky_px - zoom_base
                    # small threshold to avoid jitter
                    if abs(delta) > 4:
                        # compute smooth_delta -> scale to scroll
                        smooth_delta = delta * HZOOM_SENSITIVITY
                        # send modifier + scroll for actual app zoom
                        keyboard.press(ZOOM_MOD_KEY)
                        # vertical scroll value: positive up -> zoom in in most browsers if ctrl+scroll up zooms in.
                        mouse.scroll(0, int(smooth_delta))
                        keyboard.release(ZOOM_MOD_KEY)

                        # update zoom_base gradually to make it smoother
                        zoom_base += smooth_delta * 0.6
                        # update visual percentage
                        current_zoom_percent = clamp(int((zoom_base / 100.0) * 100 + 100), 10, 500)
                        visual_hint = f"Zoom {int(current_zoom_percent)}%"
                        last_visual_hint_time = now
            else:
                zoom_active = False

            # --- PINCH CLICK / DRAG (left click) ---
            if dist_thumb_index_px < pinch_threshold:
                # starting pinch
                if not is_pinching:
                    is_pinching = True
                    pinch_start_time = now
                    visual_hint = "Pinch start"
                    last_visual_hint_time = now
                else:
                    held = now - pinch_start_time
                    if held >= DRAG_HOLD_DURATION and not is_dragging:
                        # start drag
                        mouse.press(Button.left)
                        is_dragging = True
                        visual_hint = "Drag"
                        last_visual_hint_time = now
            else:
                # release pinch (click or end drag)
                if is_pinching:
                    if is_dragging:
                        mouse.release(Button.left)
                        is_dragging = False
                        visual_hint = "Drag End"
                        last_visual_hint_time = now
                    else:
                        # detect double click
                        if now - last_click_time <= DOUBLE_CLICK_INTERVAL:
                            mouse.click(Button.left, 2)
                            visual_hint = "Double Click"
                        else:
                            mouse.click(Button.left, 1)
                            visual_hint = "Click"
                        last_visual_hint_time = now
                        last_click_time = now
                is_pinching = False

            # --- CURSOR MAPPING ---
            # Map index normalized between calibrated corners to screen
            # linear mapping: (index.x - tl.x)/(br.x - tl.x)
            cal_tl_x, cal_tl_y = cal_tl
            cal_br_x, cal_br_y = cal_br
            # avoid division by zero
            denom_x = (cal_br_x - cal_tl_x) if abs(cal_br_x - cal_tl_x) > 1e-6 else 1e-6
            denom_y = (cal_br_y - cal_tl_y) if abs(cal_br_y - cal_tl_y) > 1e-6 else 1e-6

            normalized_x = (index.x - cal_tl_x) / denom_x
            normalized_y = (index.y - cal_tl_y) / denom_y
            # clamp
            normalized_x = clamp(normalized_x, 0.0, 1.0)
            normalized_y = clamp(normalized_y, 0.0, 1.0)

            target_x, target_y = normalized_to_screen(normalized_x, normalized_y)
            # hotspot snapping
            target_x, target_y, snapped = snap_to_hotspots(target_x, target_y)

            # exponential smoothing (alpha)
            # only move if above threshold to reduce mouse traffic
            if abs(target_x - prev_x) > CURSOR_MOVE_THRESHOLD_PX or abs(target_y - prev_y) > CURSOR_MOVE_THRESHOLD_PX:
                new_x = prev_x + (target_x - prev_x) * alpha
                new_y = prev_y + (target_y - prev_y) * alpha
                try:
                    mouse.position = (int(new_x), int(new_y))
                    prev_x, prev_y = new_x, new_y
                except Exception:
                    # if OS refused, ignore
                    pass

            # visual fingertip dot
            cv2.circle(display, (ix_px, iy_px), 10, (255, 0, 255), cv2.FILLED)

        else:
            # no hand
            if now - last_input_time > 1.0:
                visual_hint = "No hand detected"
                last_visual_hint_time = now

        # HUD - draw status, FPS, hint
        # compute FPS via pTime stored in closure
        # We'll compute simple instantaneous approx using last loop time
        if FPS_DISPLAY:
            # estimate fps by small smoothing
            # we'll compute by using time difference to previous loop iteration stored in variable
            # Using attribute on function to persist
            if not hasattr(processing_loop, "_pt"):
                processing_loop._pt = now
                processing_loop._fps = 0
            td = now - processing_loop._pt if (now - processing_loop._pt) > 1e-9 else 1e-9
            processing_loop._pt = now
            processing_loop._fps = int(0.9 * processing_loop._fps + 0.1 * (1.0 / td))
            cv2.putText(display, f"FPS: {processing_loop._fps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # gesture hint
        if visual_hint and (now - last_visual_hint_time) < 1.5:
            cv2.putText(display, visual_hint, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
        else:
            # show contextual ghost hint if user not interacting
            if now - last_input_time > HESITATION_HINT_TIME:
                cv2.putText(display, "Tip: Pinch = Click · Two-finger = Scroll · Spread = Zoom", (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 1)

        # zoom percent indicator
        cv2.putText(display, f"Zoom: {int(current_zoom_percent)}%", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # show hotspots small markers (scaled from screen to camera)
        for (hx, hy) in hotspots:
            # map from screen coords back to camera overlay for visualization
            cam_x = int((hx / screen_w) * fw)
            cam_y = int((hy / screen_h) * fh)
            cv2.circle(display, (cam_x, cam_y), 8, (0, 165, 255), 1)

        cv2.imshow("Hand-Control (backend)", display)
        key = cv2.waitKey(1) & 0xFF
        # keyboard controls
        if key == 27:
            shutdown_flag.set()
            break
        elif key == ord(' '):
            # pause / resume (just freeze cursor movement)
            # toggled by storing large last_input_time to simulate pause
            if time.time() - last_input_time < 0.5:
                # pause
                last_input_time = 0
            else:
                last_input_time = time.time()
        elif key == ord('h'):
            # save current mouse pos as hotspot
            try:
                mx, my = mouse.position
                hotspots.append((int(mx), int(my)))
                cfg = load_config()
                cfg.setdefault("hotspots", [])
                cfg["hotspots"] = hotspots
                save_config(cfg)
                print(f"Hotspot saved: {mx},{my}")
            except Exception as e:
                print("Hotspot save failed:", e)
        elif key == ord('r'):
            # recalibrate
            new_cal, new_thresh = calibrate_interactive()
            if new_cal:
                # update cal_points and threshold
                cal_tl = new_cal.get("Top-Left", (0,0))
                cal_br = new_cal.get("Bottom-Right", (1,1))
                # update config file
                cfg = load_config()
                cfg["calibration_data"] = new_cal
                cfg["pinch_threshold"] = new_thresh
                cfg.setdefault("hotspots", []).extend(hotspots)
                save_config(cfg)
                print("Recalibration done.")
        elif key == ord('m'):
            # manual pause toggle
            if time.time() - last_input_time < 0.5:
                last_input_time = 0
            else:
                last_input_time = time.time()

    # cleanup
    cv2.destroyAllWindows()

# ------------------------
# Main entry
# ------------------------
def main():
    global cap, hands, mp_draw, screen_w, screen_h, hotspots, pinch_threshold

    # Setup screen size (pynput mouse gives screen coords)
    try:
        screen_w, screen_h = pyautogui_size()
    except Exception:
        # fallback to OpenCV screen mapping heuristics
        screen_w = 1920
        screen_h = 1080

    # Setup camera
    cap_local = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2) if platform.system().lower().startswith("linux") else cv2.VideoCapture(CAMERA_INDEX)
    cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap_local.isOpened():
        print("ERROR: Camera not available.")
        return
    # assign global
    globals()['cap'] = cap_local
    try:
        cv2.startWindowThread()
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Hand-Control (backend)", cv2.WINDOW_NORMAL)
    except Exception as e:
        print("Warning: cv2.startWindowThread() failed or not required:", e)

    # MediaPipe hands
    mp_hands = mp.solutions.hands
    hands_local = mp_hands.Hands(min_detection_confidence=0.65, min_tracking_confidence=0.65, max_num_hands=1)
    globals()['hands'] = hands_local
    globals()['mp_draw'] = mp.solutions.drawing_utils

    # load config
    cfg = load_config()
    cal_data = cfg.get("calibration_data")
    pinch_threshold_cfg = cfg.get("pinch_threshold")
    if pinch_threshold_cfg:
        pinch_threshold = pinch_threshold_cfg
    if cfg.get("hotspots"):
        hotspots = cfg["hotspots"]

    # if calibration missing, run interactive
    if not cal_data:
        cal_data, computed = calibrate_interactive()
        if cal_data:
            cfg = cfg or {}
            cfg["calibration_data"] = cal_data
            cfg["pinch_threshold"] = computed
            cfg.setdefault("hotspots", []).extend(hotspots)
            save_config(cfg)
        else:
            print("Calibration aborted; exiting.")
            return

    # ensure cal keys exist
    cal_tl = cal_data.get("Top-Left", (0.0, 0.0))
    cal_br = cal_data.get("Bottom-Right", (1.0, 1.0))

    # start capture thread
    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    try:
        processing_loop(cal_data)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_flag.set()
        time.sleep(0.1)
        if cap_local:
            cap_local.release()
        cv2.destroyAllWindows()
        # persist hotspots and pinch threshold
        cfg = load_config()
        cfg["hotspots"] = hotspots
        cfg["pinch_threshold"] = pinch_threshold
        save_config(cfg)
        print("Shutdown complete.")

# helper to get screen size using cross-platform strategies
def pyautogui_size():
    # lazy import to avoid heavy import if not used
    try:
        import ctypes
        if platform.system().lower().startswith("win"):
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        elif platform.system().lower().startswith("darwin"):
            # on macOS, use Quartz
            from AppKit import NSScreen
            screen = NSScreen.mainScreen()
            frame = screen.frame()
            return int(frame.size.width), int(frame.size.height)
        else:
            # Linux, fallback to xrandr via PIL if available
            from Xlib import display as xdisplay
            d = xdisplay.Display()
            s = d.screen()
            return s.width_in_pixels, s.height_in_pixels
    except Exception:
        # fallback hardcoded
        return 1920, 1080

# Global thread handle
_hand_tracking_thread = None

def start_hand_tracking(hand_state: HandTrackingState = None):
    global _hand_tracking_thread

    if _hand_tracking_thread is None or not _hand_tracking_thread.is_alive():
        def runner():
            try:
                print("[HandTracking] Thread started")
                if hand_state is not None:
                    globals()['external_hand_state'] = hand_state

                print("[HandTracking] Initializing camera...")
                import cv2
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("[HandTracking] ERROR: Cannot open camera! Exiting thread.")
                    return
                print("[HandTracking] Camera opened successfully")

                # just capture a few frames for debug
                for i in range(5):
                    ret, frame = cap.read()
                    if not ret:
                        print("[HandTracking] ERROR: Failed to read frame")
                        break
                    print(f"[HandTracking] Captured frame {i+1}")
                    import time; time.sleep(0.1)

                cap.release()
                print("[HandTracking] Camera released")
                print("[HandTracking] Exiting thread (debug)")

            except Exception as e:
                print("Hand-tracking thread error:", e)

        _hand_tracking_thread = threading.Thread(target=runner, daemon=True)
        _hand_tracking_thread.start()
    return _hand_tracking_thread

def stop_hand_tracking():
    """
    Request the hand-tracking thread to stop and wait for it.
    """
    global _hand_tracking_thread, shutdown_flag
    shutdown_flag.set()
    if _hand_tracking_thread:
        _hand_tracking_thread.join(timeout=2.0)
        _hand_tracking_thread = None

if __name__ == "__main__":
    main()