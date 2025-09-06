import cv2
import mediapipe as mp
import pyautogui
import time
import json
import math

# Load configuration from file
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Using default values.")
    config = {
        "cursor_finger_id": 8,
        "smoothing_factor": 0.5,
        "active_zone_margin": 100,
        "fps_display_enabled": True
    }

# -- Customization Parameters loaded from config --
CURSOR_FINGER_ID = config["cursor_finger_id"]
SMOOTHING_FACTOR = config["smoothing_factor"]
ACTIVE_ZONE_MARGIN = config["active_zone_margin"]
FPS_DISPLAY_ENABLED = config["fps_display_enabled"]

# Initialize webcam and MediaPipe
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen dimensions
screen_w, screen_h = pyautogui.size()

# Initialize variables for smoothing and FPS
prev_x, prev_y = screen_w // 2, screen_h // 2
pTime = 0

# Variables for gestures
pinch_threshold = 30
last_pinch_time = 0
right_click_cooldown = 0.5

# Scrolling variables
last_scroll_y = 0
# The scroll factor now directly controls the sensitivity of the velocity-based scrolling
SCROLL_FACTOR = 0.5 

# State variables
is_pinching = False
is_dragging = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            
            index_base = hand_landmarks.landmark[5]
            middle_base = hand_landmarks.landmark[9]
            
            is_index_up = index_tip.y < index_base.y
            is_middle_up = middle_tip.y < middle_base.y

            # --- Right-Click Logic (Three-Finger Pinch) ---
            dist_thumb_index = math.sqrt((thumb_tip.x * frame_w - index_tip.x * frame_w)**2 + (thumb_tip.y * frame_h - index_tip.y * frame_h)**2)
            dist_index_middle = math.sqrt((index_tip.x * frame_w - middle_tip.x * frame_w)**2 + (index_tip.y * frame_h - middle_tip.y * frame_h)**2)
            
            if dist_thumb_index < pinch_threshold and dist_index_middle < pinch_threshold and time.time() - last_pinch_time > right_click_cooldown:
                pyautogui.rightClick()
                last_pinch_time = time.time()
                
            # --- Scrolling Logic (Two-Finger Up) ---
            if is_index_up and is_middle_up:
                current_scroll_y = index_tip.y * frame_h
                if last_scroll_y != 0:
                    # Calculate velocity and scroll
                    velocity = current_scroll_y - last_scroll_y
                    pyautogui.scroll(int(velocity * SCROLL_FACTOR))
                last_scroll_y = current_scroll_y
                
                # We use a 'continue' here to skip all other logic when in scroll mode
                continue
            else:
                last_scroll_y = 0
                
            # --- Left-Click and Drag Logic (Pinch) ---
            dist_pinch = math.sqrt((thumb_tip.x * frame_w - index_tip.x * frame_w)**2 + (thumb_tip.y * frame_h - index_tip.y * frame_h)**2)
            
            if dist_pinch < pinch_threshold:
                if not is_pinching:
                    pyautogui.click()
                    is_pinching = True
                
                if not is_dragging:
                    pyautogui.mouseDown(button='left')
                    is_dragging = True
            else:
                if is_dragging:
                    pyautogui.mouseUp(button='left')
                    is_dragging = False
                is_pinching = False

            # --- Cursor Control ---
            target_landmark = hand_landmarks.landmark[CURSOR_FINGER_ID]
            raw_x = int(target_landmark.x * frame_w)
            raw_y = int(target_landmark.y * frame_h)
            
            active_w = frame_w - 2 * ACTIVE_ZONE_MARGIN
            active_h = frame_h - 2 * ACTIVE_ZONE_MARGIN
            
            scaled_x = (raw_x - ACTIVE_ZONE_MARGIN) / active_w
            scaled_y = (raw_y - ACTIVE_ZONE_MARGIN) / active_h
            
            scaled_x = max(0, min(scaled_x, 1))
            scaled_y = max(0, min(scaled_y, 1))
            
            target_x = scaled_x * screen_w
            target_y = scaled_y * screen_h
            
            smooth_x = prev_x + (target_x - prev_x) * SMOOTHING_FACTOR
            smooth_y = prev_y + (target_y - prev_y) * SMOOTHING_FACTOR
            
            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y
            
            cv2.circle(frame, (raw_x, raw_y), 10, (255, 0, 255), cv2.FILLED)

    else:
        if is_dragging:
            pyautogui.mouseUp(button='left')
            is_dragging = False
        is_pinching = False
        last_scroll_y = 0
            
    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()