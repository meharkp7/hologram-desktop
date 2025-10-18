import cv2
import mediapipe as mp
import pyautogui
import time
import json
import math
import statistics
import os

# -------------------------
# User-tweakable defaults
# -------------------------
DEFAULT_PINCH_THRESHOLD = 30        
CLICK_COOLDOWN = 0.35               
RIGHT_CLICK_COOLDOWN = 0.5
SCROLL_FACTOR = 0.5
SMOOTHING_FACTOR = 0.5
ACTIVE_ZONE_MARGIN = 100
FPS_DISPLAY_ENABLED = True

# NEW features defaults
PINCH_DEBOUNCE_MIN = 0.05           
HESITATION_THRESHOLD = 1.2          
SNAP_RADIUS = 120                   
SNAP_STRENGTH = 0.8                 
DRAG_HOLD_DURATION = 0.25           
DEAD_ZONE_RADIUS = 10               
DOUBLE_CLICK_MAX_INTERVAL = 0.4

is_pinching = False
is_dragging = False
is_selecting = False
pinch_start_time = 0.0
last_click_time = 0.0
last_pinch_time = 0.0
last_scroll_y = 0
last_scroll_x = 0
is_paused = False
tutorial_mode_flag = False

# last input time for ghost hints
last_input_time = time.time()

def load_config(path='config.json'):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(cfg, path='config.json'):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=4)

def map_to_screen(nx, ny, cal):
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    tx = nx * screen_w
    ty = ny * screen_h
    return tx, ty

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

def calibrate_and_record_hotspots():
    """
    Enhanced calibration:
    1) Record four corner points by pinch (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    2) Collect several pinch distances samples to auto-compute pinch threshold
    3) Offer hotspot recorder: hover and press 'h' to save hotspot in screen coords
    """
    global cap, hands, mp_draw
    points = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    calibration_points = {}
    pinch_samples = []

    initial_mouse_x, initial_mouse_y = pyautogui.position()

    print("Calibration: you'll be asked to pinch at four corners of the webcam view.")
    time.sleep(1.0)

    for point_name in points:
        print(f"Calibrating: Place your index finger on the {point_name} of the webcam view and pinch to capture.")
        time.sleep(0.6)

        captured = False
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Webcam disconnected during calibration.")
                pyautogui.moveTo(initial_mouse_x, initial_mouse_y)
                cap.release()
                cv2.destroyAllWindows()
                exit()

            frame_h, frame_w, _ = frame.shape
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            cv2.putText(frame, f"Place finger on {point_name} (Pinch to capture)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]

                # draw index
                cv2.circle(frame, (int(index_tip.x * frame_w), int(index_tip.y * frame_h)), 10, (0,255,0), -1)

                dist_pinch = math.hypot((thumb_tip.x - index_tip.x) * frame_w,
                                        (thumb_tip.y - index_tip.y) * frame_h)

                # store pinch sample (even if not used immediately)
                # but only when user is attempting to pinch (visual feedback)
                if dist_pinch < DEFAULT_PINCH_THRESHOLD * 2:
                    pinch_samples.append(dist_pinch)

                if dist_pinch < DEFAULT_PINCH_THRESHOLD:
                    calibration_points[point_name] = (index_tip.x, index_tip.y)
                    print(f"Captured {point_name}: {calibration_points[point_name]}")
                    time.sleep(0.5)
                    captured = True
                    break

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("Calibration interrupted by user.")
                pyautogui.moveTo(initial_mouse_x, initial_mouse_y)
                cap.release()
                cv2.destroyAllWindows()
                exit()

        if not captured:
            print("Failed to capture point:", point_name)
            pyautogui.moveTo(initial_mouse_x, initial_mouse_y)
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # compute pinch threshold robustly
    if len(pinch_samples) >= 5:
        mean = statistics.mean(pinch_samples)
        std = statistics.stdev(pinch_samples) if len(pinch_samples) > 1 else mean * 0.12
        computed_threshold = max(10, int(mean + 1.2 * std))
        print(f"Computed pinch threshold from samples: {computed_threshold} (mean={mean:.1f}, std={std:.1f})")
    else:
        computed_threshold = DEFAULT_PINCH_THRESHOLD
        print("Insufficient pinch samples; using default threshold:", computed_threshold)

    # hotspot recorder
    hotspots = []
    print("\nHotspot recorder: hover fingertip on an app icon or button and press 'h' to record hotspots.")
    print("Press 'q' when finished recording hotspots.")
    time.sleep(0.6)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_h, frame_w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        ix = iy = None
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            ix = int(lm.landmark[8].x * frame_w)
            iy = int(lm.landmark[8].y * frame_h)
            cv2.circle(frame, (ix, iy), 8, (0,255,0), -1)
            cv2.putText(frame, "Press 'h' to save hotspot here", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, "No hand detected - press 'q' to finish", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Hotspot Recorder", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('h') and ix is not None:
            sx, sy = map_to_screen(lm.landmark[8].x, lm.landmark[8].y, calibration_points_to_screen(calibration_points, frame_w, frame_h))
            hotspots.append((sx, sy))
            print("Hotspot saved:", sx, sy)
            time.sleep(0.3)
        if k == ord('q'):
            break

    cv2.destroyWindow("Hotspot Recorder")
    pyautogui.moveTo(initial_mouse_x, initial_mouse_y)
    return calibration_points, computed_threshold, hotspots

# helper to convert normalized calibration mapping to a simple mapping - we still store normalized points
def calibration_points_to_screen(cal_pts, frame_w, frame_h):
    # return a trivial mapping for compatibility - we'll still use cal_tl and cal_br normalized in main logic
    return cal_pts

# -------------------------
# Initialize webcam and mediapipe (main)
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.65, min_tracking_confidence=0.65)  # slightly tolerant
mp_draw = mp.solutions.drawing_utils

# -------------------------
# Load config or run calibration
# -------------------------
config = load_config()
if not config or "calibration_data" not in config:
    print("No config or missing calibration data — starting enhanced calibration.")
    calibration_data, computed_thresh, hotspots = calibrate_and_record_hotspots()
    config = config or {}
    config["calibration_data"] = calibration_data
    config["pinch_threshold"] = computed_thresh
    config["hotspots"] = hotspots
    config.setdefault("seen_tutorials", {"work": False, "presentation": False, "media": False})
    config["tutorial_completed"] = False
    save_config(config)
    print("Calibration complete. Config saved to config.json")

# Make sure keys exist
config.setdefault("pinch_threshold", config.get("pinch_threshold", DEFAULT_PINCH_THRESHOLD))
config.setdefault("hotspots", config.get("hotspots", []))
config.setdefault("seen_tutorials", config.get("seen_tutorials", {"work": False, "presentation": False, "media": False}))
config.setdefault("tutorial_completed", config.get("tutorial_completed", False))

# apply parameters to runtime
PINCH_THRESHOLD = int(config["pinch_threshold"])
hotspots = config.get("hotspots", [])

CURSOR_FINGER_ID = config.get("cursor_finger_id", 8)
SMOOTHING_FACTOR = config.get("smoothing_factor", SMOOTHING_FACTOR)
ACTIVE_ZONE_MARGIN = config.get("active_zone_margin", ACTIVE_ZONE_MARGIN)
FPS_DISPLAY_ENABLED = config.get("fps_display_enabled", FPS_DISPLAY_ENABLED)

cal_tl_x, cal_tl_y = config["calibration_data"]["Top-Left"]
cal_br_x, cal_br_y = config["calibration_data"]["Bottom-Right"]

# Get screen size
screen_w, screen_h = pyautogui.size()

# smoothing state
prev_x, prev_y = screen_w // 2, screen_h // 2
pTime = 0.0

# function to show quick tutorial (core gestures) - run once unless reset
def run_tutorial_once():
    global config
    if config.get("tutorial_completed", False):
        return
    steps = [
        "Tutorial: Move pointer with your index finger",
        "Pinch (thumb+index) to click. Hold pinch to drag.",
        "Two-finger swipe (index+middle) to scroll."
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

# Run tutorial if first-run
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
    movement_detected = False
    now = time.time()

    if results.multi_hand_landmarks:
        is_paused = False
        # update last_input_time as there's a hand visible (so not paused)
        last_input_time = now

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # fingertips
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # bases for finger-up heuristics
            index_base = hand_landmarks.landmark[5]
            middle_base = hand_landmarks.landmark[9]
            ring_base = hand_landmarks.landmark[13]
            pinky_base = hand_landmarks.landmark[17]

            is_index_up = index_tip.y < index_base.y
            is_middle_up = middle_tip.y < middle_base.y
            is_ring_up = ring_tip.y < ring_base.y
            is_pinky_up = pinky_tip.y < pinky_base.y

            is_open_palm = is_index_up and is_middle_up and is_ring_up and is_pinky_up

            # distances (camera pixels)
            dist_thumb_index = math.hypot((thumb_tip.x - index_tip.x) * frame_w,
                                          (thumb_tip.y - index_tip.y) * frame_h)
            dist_index_middle = math.hypot((index_tip.x - middle_tip.x) * frame_w,
                                           (index_tip.y - middle_tip.y) * frame_h)

            # RIGHT CLICK (three-finger pinch)
            if dist_thumb_index < PINCH_THRESHOLD and dist_index_middle < PINCH_THRESHOLD and (now - last_click_time) > RIGHT_CLICK_COOLDOWN:
                pyautogui.rightClick()
                last_click_time = now
                gesture_text = "Right Click"
                last_input_time = now

            # SCROLL (two-finger up)
            if is_index_up and is_middle_up and not is_ring_up and not is_pinky_up:
                current_scroll_y = index_tip.y * frame_h
                current_scroll_x = index_tip.x * frame_w

                if last_scroll_y != 0 or last_scroll_x != 0:
                    scroll_delta_y = last_scroll_y - current_scroll_y
                    scroll_delta_x = last_scroll_x - current_scroll_x

                    if abs(scroll_delta_y) > abs(scroll_delta_x):
                        pyautogui.scroll(int(scroll_delta_y * SCROLL_FACTOR))
                        gesture_text = "Scroll"
                    else:
                        pyautogui.hscroll(int(scroll_delta_x * -SCROLL_FACTOR))
                        gesture_text = "H-Scroll"

                    last_input_time = now

                last_scroll_y = current_scroll_y
                last_scroll_x = current_scroll_x

                # show scroll indicator circle
                cv2.circle(frame, (int(index_tip.x * frame_w), int(index_tip.y * frame_h)), 14, (0, 255, 255), 2)
                # skip other controls while scrolling for stability
                continue
            else:
                last_scroll_y = 0
                last_scroll_x = 0

            # PINCH (debounced) and DRAG
            if dist_thumb_index < PINCH_THRESHOLD:
                if not is_pinching:
                    is_pinching = True
                    pinch_start_time = now
                    gesture_text = "Pinch (start)"
                else:
                    held = now - pinch_start_time
                    # Confirm pinch after debounce
                    if held >= PINCH_DEBOUNCE_MIN:
                        last_input_time = now
                        # double-click detection
                        if (now - last_pinch_time) < DOUBLE_CLICK_MAX_INTERVAL and (now - last_click_time) > CLICK_COOLDOWN:
                            pyautogui.doubleClick()
                            last_click_time = now
                            last_pinch_time = 0.0
                            gesture_text = "Double Click"
                            is_pinching = False
                        else:
                            # drag start if held long enough
                            if held >= DRAG_HOLD_DURATION and not is_dragging:
                                pyautogui.mouseDown(button='left')
                                is_dragging = True
                                is_selecting = True
                                gesture_text = "Drag Start"
                            elif not is_dragging and (now - last_click_time) > CLICK_COOLDOWN and held < DRAG_HOLD_DURATION:
                                # quick click (short pinch)
                                pyautogui.click()
                                last_click_time = now
                                gesture_text = "Click"
                last_pinch_time = now
            else:
                # pinch released
                if is_dragging:
                    pyautogui.mouseUp(button='left')
                    is_dragging = False
                    is_selecting = False
                    gesture_text = "Drag End"
                is_pinching = False

            # CURSOR POSITIONING (index finger used for pointer)
            normalized_x = (index_tip.x - cal_tl_x) / (cal_br_x - cal_tl_x)
            normalized_y = (index_tip.y - cal_tl_y) / (cal_br_y - cal_tl_y)
            target_x, target_y = map_to_screen(normalized_x, normalized_y, config["calibration_data"])

            # Snap to hotspot if near
            target_x_snapped, target_y_snapped, snapped = snap_to_hotspot(target_x, target_y, hotspots)
            if snapped:
                # visual halo around hotspot
                cv2.circle(frame, (int((target_x_snapped / screen_w) * frame_w), int((target_y_snapped / screen_h) * frame_h)), 22, (0,200,255), 2)
                gesture_text = "Near Hotspot"

            # smoothing: more responsive when dragging
            smoothing = 0.18 if is_dragging else SMOOTHING_FACTOR

            # apply dead-zone to avoid micro-movements
            if math.hypot(target_x_snapped - prev_x, target_y_snapped - prev_y) > DEAD_ZONE_RADIUS:
                prev_x += (target_x_snapped - prev_x) * smoothing
                prev_y += (target_y_snapped - prev_y) * smoothing
                pyautogui.moveTo(prev_x, prev_y)
                movement_detected = True
                last_input_time = now

            # Visual cursor color-coded
            color = (255, 0, 255)  # purple default (BGR)
            if is_dragging:
                color = (0, 165, 255)  # orange-ish for dragging
            elif is_pinching:
                color = (0, 255, 0)  # green while pinching
            cv2.circle(frame, (int(prev_x / screen_w * frame_w), int(prev_y / screen_h * frame_h)), 12, color, cv2.FILLED)

            # pinch progress bar
            pinch_bar_x = 20
            pinch_bar_y = 80
            cv2.rectangle(frame, (pinch_bar_x, pinch_bar_y), (pinch_bar_x + 120, pinch_bar_y + 12), (50, 50, 50), cv2.FILLED)
            norm_pinch = max(0, min(1, 1 - (dist_thumb_index / (PINCH_THRESHOLD * 2))))
            cv2.rectangle(frame, (pinch_bar_x, pinch_bar_y), (pinch_bar_x + int(120 * norm_pinch), pinch_bar_y + 12), (0, 200, 0), cv2.FILLED)

    else:
        # No hand detected
        if is_selecting or is_dragging:
            pyautogui.mouseUp(button='left')
        is_selecting = False
        is_dragging = False
        is_pinching = False
        last_scroll_y = last_scroll_x = 0
        is_paused = True

    # Ghost hints on hesitation
    if time.time() - last_input_time > HESITATION_THRESHOLD:
        # show subtle contextual hint
        show_hint(frame, "Tip: Pinch to click · Two-finger swipe to scroll")
    else:
        # not showing hint
        pass

    # HUD & FPS
    cTime = time.time()
    fps = int(1.0 / (cTime - pTime)) if pTime else 0
    pTime = cTime

    status_text = "Paused" if is_paused else "Active"
    cv2.putText(frame, f'Status: {status_text}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if is_paused else (0, 255, 0), 2)
    if FPS_DISPLAY_ENABLED:
        cv2.putText(frame, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f'Gesture: {gesture_text}', (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

    cv2.imshow("Hand Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC exit
        break
    if key == ord(' '):  # space toggles pause
        is_paused = not is_paused

# end main loop
cap.release()
cv2.destroyAllWindows()
# save updated config (hotspots / pinch threshold persisted earlier at calibration)
config["pinch_threshold"] = PINCH_THRESHOLD
config["hotspots"] = hotspots
save_config(config)
print("Exiting. Config saved.")
