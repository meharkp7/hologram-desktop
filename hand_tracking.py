import cv2
import mediapipe as mp
import pyautogui
import time
import json
import math

# --- State variables (must be initialized before any functions use them) ---
pinch_threshold = 30
click_cooldown = 0.5
right_click_cooldown = 0.5
SCROLL_FACTOR = 0.5
SMOOTHING_FACTOR = 0.5
ACTIVE_ZONE_MARGIN = 100
FPS_DISPLAY_ENABLED = True
is_pinching = False
is_dragging = False
is_selecting = False
drag_start_time = 0
drag_hold_duration = 0.25
last_pinch_time = 0
last_scroll_y = 0
last_scroll_x = 0
is_paused = False

# --- Calibration Function ---
def calibrate():
    points = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    calibration_points = {}
    
    initial_mouse_x, initial_mouse_y = pyautogui.position()

    for point_name in points:
        print(f"Calibrating: Place your index finger on the {point_name} of the webcam view and pinch your thumb and index finger to capture.")
        
        time.sleep(1) 

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_h, frame_w, _ = frame.shape
            frame = cv2.flip(frame, 1)

            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Place finger on {point_name} (Pinch to capture)"
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, 1, 2)
            
            text_x = 20
            text_y = 50
            
            cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), 
                          (text_x + text_width + 5, text_y + baseline + 5), 
                          (0, 0, 0), cv2.FILLED)
            
            cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                index_tip = hand_landmarks.landmark[8]
                
                cv2.circle(frame, (int(index_tip.x * frame_w), int(index_tip.y * frame_h)), 10, (0, 255, 0), cv2.FILLED)
                
                thumb_tip = hand_landmarks.landmark[4]
                dist_pinch = math.sqrt((thumb_tip.x * frame_w - index_tip.x * frame_w)**2 + (thumb_tip.y * frame_h - index_tip.y * frame_h)**2)
                
                if dist_pinch < pinch_threshold:
                    calibration_points[point_name] = (index_tip.x, index_tip.y)
                    print(f"Captured {point_name}: {calibration_points[point_name]}")
                    time.sleep(1)
                    break
            
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("Calibration interrupted by user.")
                pyautogui.moveTo(initial_mouse_x, initial_mouse_y)
                cap.release()
                cv2.destroyAllWindows()
                exit()
        
        if not ret:
            print("Webcam disconnected during calibration.")
            pyautogui.moveTo(initial_mouse_x, initial_mouse_y)
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    cv2.destroyAllWindows()
    pyautogui.moveTo(initial_mouse_x, initial_mouse_y)
    return calibration_points

# --- Main Program ---
# Initialize webcam and MediaPipe
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load configuration from file
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    if "calibration_data" not in config:
        print("Calibration data not found. Starting calibration...")
        raise FileNotFoundError
except FileNotFoundError:
    config = {
        "cursor_finger_id": 8,
        "smoothing_factor": SMOOTHING_FACTOR,
        "active_zone_margin": ACTIVE_ZONE_MARGIN,
        "fps_display_enabled": FPS_DISPLAY_ENABLED
    }
    calibration_data = calibrate() 
    config["calibration_data"] = calibration_data
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("Calibration complete. Data saved.")
    
# -- Update parameters from config --
CURSOR_FINGER_ID = config["cursor_finger_id"]
SMOOTHING_FACTOR = config["smoothing_factor"]
ACTIVE_ZONE_MARGIN = config["active_zone_margin"]
FPS_DISPLAY_ENABLED = config["fps_display_enabled"]

# Get calibration data
cal_tl_x, cal_tl_y = config["calibration_data"]["Top-Left"]
cal_br_x, cal_br_y = config["calibration_data"]["Bottom-Right"]

# Get screen dimensions
screen_w, screen_h = pyautogui.size()

# Initialize variables for smoothing and FPS
prev_x, prev_y = screen_w // 2, screen_h // 2
pTime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        is_paused = False
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            index_base = hand_landmarks.landmark[5]
            middle_base = hand_landmarks.landmark[9]
            ring_base = hand_landmarks.landmark[13]
            pinky_base = hand_landmarks.landmark[17]

            is_index_up = index_tip.y < index_base.y
            is_middle_up = middle_tip.y < middle_base.y
            is_ring_up = ring_tip.y < ring_base.y
            is_pinky_up = pinky_tip.y < pinky_base.y
            
            is_fist = not is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up
            is_open_palm = is_index_up and is_middle_up and is_ring_up and is_pinky_up

            # All gestures and pyautogui commands are now wrapped in a check for "not paused"
            if not is_paused:
                # --- Right-Click Logic (Three-Finger Pinch) ---
                dist_thumb_index = math.sqrt((thumb_tip.x * frame_w - index_tip.x * frame_w)**2 + (thumb_tip.y * frame_h - index_tip.y * frame_h)**2)
                dist_index_middle = math.sqrt((index_tip.x * frame_w - middle_tip.x * frame_w)**2 + (index_tip.y * frame_h - middle_tip.y * frame_h)**2)
                
                if dist_thumb_index < pinch_threshold and dist_index_middle < pinch_threshold and time.time() - last_pinch_time > right_click_cooldown:
                    pyautogui.rightClick()
                    last_pinch_time = time.time()

                # --- Scrolling Logic (Two-Finger Up) ---
                if is_index_up and is_middle_up and not is_ring_up and not is_pinky_up:
                    current_scroll_y = index_tip.y * frame_h
                    current_scroll_x = index_tip.x * frame_w

                    if last_scroll_y != 0 or last_scroll_x != 0:
                        scroll_delta_y = last_scroll_y - current_scroll_y
                        scroll_delta_x = last_scroll_x - current_scroll_x

                        if abs(scroll_delta_y) > abs(scroll_delta_x):
                            pyautogui.scroll(int(scroll_delta_y * SCROLL_FACTOR))
                        else:
                            pyautogui.hscroll(int(scroll_delta_x * -SCROLL_FACTOR))

                    last_scroll_y = current_scroll_y
                    last_scroll_x = current_scroll_x
                    
                    continue
                else:
                    last_scroll_y = 0
                    last_scroll_x = 0
                
                # --- Left-Click Logic (Quick Pinch) ---
                dist_pinch = math.sqrt((thumb_tip.x * frame_w - index_tip.x * frame_w)**2 + (thumb_tip.y * frame_h - index_tip.y * frame_h)**2)
                
                if dist_pinch < pinch_threshold and (time.time() - last_pinch_time) > click_cooldown and not is_pinching:
                    pyautogui.click()
                    is_pinching = True
                elif dist_pinch >= pinch_threshold:
                    is_pinching = False
                
                # --- Dragging & Selection Logic ---
                if is_fist:
                    if not is_selecting:
                        pyautogui.mouseDown(button='left')
                        is_selecting = True
                        drag_start_time = time.time()
                    
                    if (time.time() - drag_start_time) > drag_hold_duration and not is_dragging:
                        is_dragging = True
                
                if is_open_palm and is_dragging:
                    pyautogui.mouseUp(button='left')
                    is_selecting = False
                    is_dragging = False
                    drag_start_time = 0
                    
                if not is_fist and is_selecting:
                    pyautogui.mouseUp(button='left')
                    is_selecting = False
                    
                # --- Cursor Control ---
                normalized_x = (index_tip.x - cal_tl_x) / (cal_br_x - cal_tl_x)
                normalized_y = (index_tip.y - cal_tl_y) / (cal_br_y - cal_tl_y)
                
                if is_dragging:
                    drag_landmark = hand_landmarks.landmark[9]
                    normalized_drag_x = (drag_landmark.x - cal_tl_x) / (cal_br_x - cal_tl_x)
                    normalized_drag_y = (drag_landmark.y - cal_tl_y) / (cal_br_y - cal_tl_y)
                    
                    target_x = normalized_drag_x * screen_w
                    target_y = normalized_drag_y * screen_h
                else:
                    target_x = normalized_x * screen_w
                    target_y = normalized_y * screen_h
                
                smooth_x = prev_x + (target_x - prev_x) * SMOOTHING_FACTOR
                smooth_y = prev_y + (target_y - prev_y) * SMOOTHING_FACTOR
                
                pyautogui.moveTo(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y
                
                # Get the raw coordinates for the circle after all calculations
                if is_dragging:
                    raw_x = int(hand_landmarks.landmark[9].x * frame_w)
                    raw_y = int(hand_landmarks.landmark[9].y * frame_h)
                else:
                    raw_x = int(index_tip.x * frame_w)
                    raw_y = int(index_tip.y * frame_h)

                cv2.circle(frame, (raw_x, raw_y), 10, (255, 0, 255), cv2.FILLED)

    else:
        is_paused = True
        
        # Reset any active states to prevent ghost inputs
        if is_selecting:
            pyautogui.mouseUp(button='left')
            is_selecting = False
        if is_dragging:
            pyautogui.mouseUp(button='left')
            is_dragging = False
        is_pinching = False
        drag_start_time = 0
        last_scroll_y = 0
        last_scroll_x = 0
        
    # Display FPS and status
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    status_text = "Status: Paused" if is_paused else "Status: Active"
    cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_paused else (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()