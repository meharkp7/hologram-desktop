# backend/main.py
import cv2
import mediapipe as mp
import pyautogui
import time
import json
import math
import os
import statistics

# -------------------------
# Defaults & tunables
# -------------------------
DEFAULT_PINCH_THRESHOLD = 30
PINCH_DEBOUNCE_MIN = 0.05
CLICK_COOLDOWN = 0.35
RIGHT_CLICK_COOLDOWN = 0.5
DOUBLE_CLICK_MAX_INTERVAL = 0.4
DRAG_HOLD_DURATION = 0.25
SCROLL_FACTOR = 0.5
SMOOTHING_FACTOR = 0.5
ACTIVE_ZONE_MARGIN = 100
HESITATION_THRESHOLD = 1.2
DEAD_ZONE_RADIUS = 10
FPS_DISPLAY_ENABLED = True
SNAP_RADIUS = 120
SNAP_STRENGTH = 0.8

# runtime states
is_pinching = False
is_dragging = False
is_selecting = False
pinch_start_time = 0.0
last_click_time = 0.0
last_pinch_time = 0.0
last_scroll_y = 0
last_scroll_x = 0
is_paused = False
last_input_time = time.time()
ghost_text = ""
last_ghost_time = 0
GHOST_DISPLAY_TIME = 1.5

# -------------------------
# Helpers
# -------------------------
def load_config(path='backend/config.json'):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(cfg, path='backend/config.json'):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=4)

def map_to_screen(nx, ny):
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    return nx * screen_w, ny * screen_h

def show_hint(frame, text):
    fh, fw, _ = frame.shape
    cv2.putText(frame, text, (30, fh - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,255), 2)

def snap_to_hotspot(x, y, hotspots):
    if not hotspots:
        return x, y, False
    best = min(hotspots, key=lambda p: math.hypot(p[0]-x, p[1]-y))
    d = math.hypot(best[0]-x, best[1]-y)
    if d < SNAP_RADIUS:
        nx = x + (best[0]-x) * SNAP_STRENGTH
        ny = y + (best[1]-y) * SNAP_STRENGTH
        return nx, ny, True
    return x, y, False

# -------------------------
# Initialize Webcam & MediaPipe
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.65, min_tracking_confidence=0.65)
mp_draw = mp.solutions.drawing_utils

# -------------------------
# Load config or calibrate
# -------------------------
config = load_config()
if not config or "calibration_data" not in config:
    import backend.calibration as calibration
    print("Calibration data not found. Running calibration...")
    calibration_data = calibration.calibrate(cap, hands)
    config = {"calibration_data": calibration_data}
    save_config(config)
    print("Calibration complete.")

PINCH_THRESHOLD = int(config.get("pinch_threshold", DEFAULT_PINCH_THRESHOLD))
hotspots = config.get("hotspots", [])
cal_tl_x, cal_tl_y = config["calibration_data"]["Top-Left"]
cal_br_x, cal_br_y = config["calibration_data"]["Bottom-Right"]

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = screen_w // 2, screen_h // 2
pTime = 0.0

# -------------------------
# Tutorial (phase-1 basic)
# -------------------------
def run_tutorial_once():
    if config.get("tutorial_completed", False):
        return
    steps = [
        "Move pointer with your index finger",
        "Pinch (thumb+index) to click or drag",
        "Two-finger swipe to scroll"
    ]
    for text in steps:
        t0 = time.time()
        while time.time() - t0 < 2.0:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240,240,200), 3)
            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    config["tutorial_completed"] = True
    save_config(config)
    time.sleep(0.4)

run_tutorial_once()

# -------------------------
# Main Loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = "None"
    now = time.time()
    movement_detected = False

    if results.multi_hand_landmarks and not is_paused:
        last_input_time = now
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        index_base = hand_landmarks.landmark[5]
        middle_base = hand_landmarks.landmark[9]

        is_index_up = index_tip.y < index_base.y
        is_middle_up = middle_tip.y < middle_base.y

        # DISTANCES
        dist_thumb_index = math.hypot((thumb_tip.x - index_tip.x)*frame_w, (thumb_tip.y - index_tip.y)*frame_h)
        dist_index_middle = math.hypot((index_tip.x - middle_tip.x)*frame_w, (index_tip.y - middle_tip.y)*frame_h)

        # Right-click (three-finger pinch simplified)
        if dist_thumb_index < PINCH_THRESHOLD and dist_index_middle < PINCH_THRESHOLD and (now - last_click_time) > RIGHT_CLICK_COOLDOWN:
            pyautogui.rightClick()
            gesture_text = "Right Click"
            last_click_time = now

        # Scroll
        if is_index_up and is_middle_up:
            current_scroll_y = index_tip.y * frame_h
            if last_scroll_y != 0:
                pyautogui.scroll(int((last_scroll_y - current_scroll_y) * SCROLL_FACTOR))
                gesture_text = "Scroll"
            last_scroll_y = current_scroll_y
        else:
            last_scroll_y = 0

        # Pinch click / drag
        if dist_thumb_index < PINCH_THRESHOLD:
            if not is_pinching:
                is_pinching = True
                pinch_start_time = now
                gesture_text = "Pinch Start"
            else:
                held = now - pinch_start_time
                if held >= PINCH_DEBOUNCE_MIN:
                    if (now - last_pinch_time) < DOUBLE_CLICK_MAX_INTERVAL and (now - last_click_time) > CLICK_COOLDOWN:
                        pyautogui.doubleClick()
                        gesture_text = "Double Click"
                        last_click_time = now
                        last_pinch_time = 0.0
                        is_pinching = False
                    else:
                        if held >= DRAG_HOLD_DURATION and not is_dragging:
                            pyautogui.mouseDown(button='left')
                            is_dragging = True
                            is_selecting = True
                            gesture_text = "Drag Start"
                        elif not is_dragging and (now - last_click_time) > CLICK_COOLDOWN:
                            pyautogui.click()
                            gesture_text = "Click"
                            last_click_time = now
            last_pinch_time = now
        else:
            if is_dragging:
                pyautogui.mouseUp(button='left')
                is_dragging = False
                is_selecting = False
                gesture_text = "Drag End"
            is_pinching = False

        # Cursor
        normalized_x = (index_tip.x - cal_tl_x) / (cal_br_x - cal_tl_x)
        normalized_y = (index_tip.y - cal_tl_y) / (cal_br_y - cal_tl_y)
        target_x, target_y = map_to_screen(normalized_x, normalized_y)

        # Snap to hotspots
        target_x, target_y, snapped = snap_to_hotspot(target_x, target_y, hotspots)
        smoothing = 0.18 if is_dragging else SMOOTHING_FACTOR
        if math.hypot(target_x - prev_x, target_y - prev_y) > DEAD_ZONE_RADIUS:
            prev_x += (target_x - prev_x) * smoothing
            prev_y += (target_y - prev_y) * smoothing
            pyautogui.moveTo(prev_x, prev_y)
            movement_detected = True
            last_input_time = now

        color = (255, 0, 255)
        if is_dragging: color = (0,165,255)
        elif is_pinching: color = (0,255,0)
        cv2.circle(frame, (int(prev_x/screen_w*frame_w), int(prev_y/screen_h*frame_h)), 12, color, cv2.FILLED)

    else:
        # No hand
        if is_dragging or is_selecting:
            pyautogui.mouseUp(button='left')
        is_dragging = is_selecting = is_pinching = False
        last_scroll_y = 0
        is_paused = True

    # Ghost hints on hesitation
    if time.time() - last_input_time > HESITATION_THRESHOLD:
        show_hint(frame, "Tip: Pinch to click · Two-finger swipe to scroll")

    # HUD
    cTime = time.time()
    fps = int(1.0 / (cTime - pTime)) if pTime else 0
    pTime = cTime

    status_text = "Paused" if is_paused else "Active"
    cv2.putText(frame, f'Status: {status_text}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,0,255) if is_paused else (0,255,0), 2)
    if FPS_DISPLAY_ENABLED:
        cv2.putText(frame, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f'Gesture: {gesture_text}', (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

    cv2.imshow("Hand Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    if key == ord(' '): is_paused = not is_paused

# Cleanup
cap.release()
cv2.destroyAllWindows()
save_config(config)
print("Exiting. Config saved.")