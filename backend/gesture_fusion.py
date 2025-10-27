#!/usr/bin/env python3
import cv2
import threading
import time
import platform
from hand_tracking_backend import (
    capture_thread,
    load_config,
    save_config,
    calibrate_interactive,
    hotspots,
    pinch_threshold,
    _latest_frame,
    screen_w,
    screen_h,
    last_hand_position,
    is_pinching
)
import hand_tracking_backend as htb
from ai_keyboard import AirKeyboard

keyboard = AirKeyboard()
keyboard_visible = False

# --- Camera detection ---
def detect_camera():
    system = platform.system().lower()
    backends = {
        "darwin": [cv2.CAP_AVFOUNDATION, cv2.CAP_QT, cv2.CAP_ANY],
        "linux": [cv2.CAP_V4L2, cv2.CAP_ANY],
        "windows": [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    }.get(system, [cv2.CAP_ANY])

    for backend in backends:
        for idx in range(5):
            cap_test = cv2.VideoCapture(idx, backend)
            if cap_test.isOpened():
                cap_test.release()
                return idx, backend
    return None, None

# --- Draw overlays ---
def draw_overlay(frame):
    h, w, _ = frame.shape
    # Draw hotspots
    for hx, hy, hr in hotspots:
        cv2.circle(frame, (int(hx*w), int(hy*h)), int(hr*min(w,h)), (0,255,0), 2)
    # Draw pinch threshold
    cv2.line(frame, (0,int(pinch_threshold*h)), (w,int(pinch_threshold*h)), (255,0,0), 1)
    # Draw keyboard toggle button bottom-right
    btn_w, btn_h = 120, 50
    cv2.rectangle(frame, (w-btn_w-20,h-btn_h-20), (w-20,h-20), (50,50,50), -1)
    cv2.putText(frame, "Keyboard", (w-btn_w-15,h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

# --- Toggle button click ---
def check_keyboard_toggle(x, y, frame_w, frame_h):
    global keyboard_visible
    btn_w, btn_h = 120, 50
    if frame_w-btn_w-20 <= x <= frame_w-20 and frame_h-btn_h-20 <= y <= frame_h-20:
        keyboard_visible = not keyboard_visible

# --- Draw AI keyboard ---
def draw_keyboard(frame):
    keys = list("QWERTYUIOPASDFGHJKLZXCVBNM")
    key_w, key_h = 60, 60
    start_x, start_y = 50, frame.shape[0]-250
    for i, key in enumerate(keys):
        row, col = 0, i
        if i >= 10 and i < 19:
            row, col = 1, i-10
        elif i >= 19:
            row, col = 2, i-19
        x, y = start_x + col*(key_w+10), start_y + row*(key_h+10)
        cv2.rectangle(frame, (x,y), (x+key_w,y+key_h), (100,100,100), -1)
        cv2.putText(frame, key, (x+15,y+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# --- Check finger press ---
def check_key_press(finger_pos, frame):
    if finger_pos is None:
        return None
    x, y = finger_pos
    keys = list("QWERTYUIOPASDFGHJKLZXCVBNM")
    key_w, key_h = 60, 60
    start_x, start_y = 50, frame.shape[0]-250
    for i, key in enumerate(keys):
        row, col = 0, i
        if i >= 10 and i < 19:
            row, col = 1, i-10
        elif i >= 19:
            row, col = 2, i-19
        kx, ky = start_x + col*(key_w+10), start_y + row*(key_h+10)
        if kx <= x <= kx+key_w and ky <= y <= ky+key_h:
            return key
    return None

# --- Main ---
def main():
    global keyboard_visible, keyboard

    # Screen size
    try:
        from hand_tracking_backend import pyautogui_size
        w, h = pyautogui_size()
        htb.screen_w, htb.screen_h = w, h
    except Exception:
        htb.screen_w, htb.screen_h = 1920, 1080

    # Detect camera
    camera_index, backend = detect_camera()
    if camera_index is None:
        print("ERROR: No camera found.")
        return
    print(f"Using camera {camera_index} with backend {backend}")
    htb.cap = cv2.VideoCapture(camera_index, backend)
    htb.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    htb.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not htb.cap.isOpened():
        print("ERROR: Camera could not be opened.")
        return

    # OpenCV window
    cv2.startWindowThread()
    cv2.namedWindow("Hand-Control Overlay", cv2.WINDOW_NORMAL)

    # Load config
    cfg = load_config()
    cal_data = cfg.get("calibration_data")
    if cfg.get("pinch_threshold"):
        htb.pinch_threshold = cfg["pinch_threshold"]
    if cfg.get("hotspots"):
        htb.hotspots.extend(cfg["hotspots"])

    # Calibration
    if not cal_data:
        cal_data, computed = calibrate_interactive()
        if cal_data:
            cfg["calibration_data"] = cal_data
            cfg["pinch_threshold"] = computed
            cfg.setdefault("hotspots", []).extend(hotspots)
            save_config(cfg)
        else:
            print("Calibration aborted; exiting.")
            return

    # Start capture thread
    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    # Main loop
    try:
        while True:
            if _latest_frame is not None:
                frame = _latest_frame.copy()
                h, w, _ = frame.shape

                # Draw gestures & toggle button
                draw_overlay(frame)

                # Mouse callback
                def mouse_cb(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        check_keyboard_toggle(x, y, w, h)
                cv2.setMouseCallback("Hand-Control Overlay", mouse_cb)

                # Keyboard overlay
                if keyboard_visible:
                    draw_keyboard(frame)
                    finger = last_hand_position()
                    pinch = is_pinching()
                    if finger and pinch:
                        key = check_key_press(finger, frame)
                        if key:
                            keyboard.key_pressed(key)

                    # Display typed text
                    cv2.rectangle(frame, (50,50), (w-50,120), (30,30,30), -1)
                    cv2.putText(frame, keyboard.typed_text, (60,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

                cv2.imshow("Hand-Control Overlay", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if htb.cap:
            htb.cap.release()
        cv2.destroyAllWindows()
        cfg["hotspots"] = htb.hotspots
        cfg["pinch_threshold"] = htb.pinch_threshold
        save_config(cfg)
        print("Shutdown complete.")

if __name__ == "__main__":
    main()