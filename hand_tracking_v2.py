# hand_tracking_final_v4.py
import cv2
import mediapipe as mp
import pyautogui
import time
import json
import math
import statistics
import os

# -----------------------
# User-tweakable defaults
# -----------------------
DEFAULT_PINCH_THRESHOLD = 30
CLICK_COOLDOWN = 0.35
RIGHT_CLICK_COOLDOWN = 0.5
SCROLL_FACTOR = 300
HZOOM_FACTOR = 3.0
SMOOTHING_FACTOR = 0.5
SNAP_RADIUS = 120
SNAP_STRENGTH = 0.8
DRAG_HOLD_DURATION = 0.25
DOUBLE_CLICK_MAX_INTERVAL = 0.4
FPS_DISPLAY_ENABLED = True

# -----------------------
# Runtime state
# -----------------------
is_pinching = False
is_dragging = False
last_click_time = 0.0
last_right_click_time = 0.0
last_scroll_pos = None
is_paused = False
last_input_time = time.time()
hotspots = []
zoom_base = 0
zoom_active = False

# -----------------------
# Helper functions
# -----------------------
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

def map_to_screen(nx, ny):
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    return nx * screen_w, ny * screen_h

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

# -----------------------
# Calibration
# -----------------------
def calibrate():
    global cap, hands, mp_draw
    points = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    cal_points = {}
    pinch_samples = []
    print("Calibration: Place your fingertip at 4 corners and pinch.")

    for point in points:
        print(f"Place index finger at {point} and pinch...")
        captured = False
        while not captured:
            ret, frame = cap.read()
            if not ret:
                print("Webcam disconnected during calibration.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            fh, fw, _ = frame.shape
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            cv2.putText(frame, f"Pinch at {point}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                index_tip = lm.landmark[8]
                thumb_tip = lm.landmark[4]

                cv2.circle(frame, (int(index_tip.x*fw), int(index_tip.y*fh)), 10, (0,255,0), -1)

                dist_pinch = math.hypot((thumb_tip.x-index_tip.x)*fw, (thumb_tip.y-index_tip.y)*fh)
                if dist_pinch < DEFAULT_PINCH_THRESHOLD*2:
                    pinch_samples.append(dist_pinch)

                if dist_pinch < DEFAULT_PINCH_THRESHOLD:
                    cal_points[point] = (index_tip.x, index_tip.y)
                    print(f"Captured {point}: {cal_points[point]}")
                    time.sleep(0.5)
                    captured = True

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()
    cv2.destroyWindow("Calibration")

    if len(pinch_samples) >= 5:
        mean = statistics.mean(pinch_samples)
        std = statistics.stdev(pinch_samples) if len(pinch_samples) > 1 else mean*0.12
        threshold = max(10,int(mean+1.2*std))
    else:
        threshold = DEFAULT_PINCH_THRESHOLD
    print(f"Computed pinch threshold: {threshold}")
    return cal_points, threshold

# -----------------------
# Initialize
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.65, min_tracking_confidence=0.65)
mp_draw = mp.solutions.drawing_utils

config = load_config()
if not config or "calibration_data" not in config:
    cal_data, pinch_thresh = calibrate()
    config = {"calibration_data": cal_data, "pinch_threshold": pinch_thresh, "hotspots":[]}
    save_config(config)

PINCH_THRESHOLD = config["pinch_threshold"]
hotspots = config.get("hotspots", [])

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = screen_w//2, screen_h//2
pTime = 0.0

# -----------------------
# Main Loop
# -----------------------
print("Starting main loop. ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fh, fw, _ = frame.shape
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    gesture_text = "None"
    now = time.time()

    if res.multi_hand_landmarks:
        last_input_time = now
        for lm in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            thumb_tip = lm.landmark[4]
            index_tip = lm.landmark[8]
            middle_tip = lm.landmark[12]
            ring_tip = lm.landmark[16]
            pinky_tip = lm.landmark[20]

            # -------------------
            # Fingers up
            fingers_up = [
                index_tip.y < lm.landmark[5].y,
                middle_tip.y < lm.landmark[9].y,
                ring_tip.y < lm.landmark[13].y,
                pinky_tip.y < lm.landmark[17].y
            ]

            # Distances
            dist_thumb_index = math.hypot((thumb_tip.x-index_tip.x)*fw, (thumb_tip.y-index_tip.y)*fh)
            dist_thumb_pinky = math.hypot((thumb_tip.x-pinky_tip.x)*fw, (thumb_tip.y-pinky_tip.y)*fh)

            # -------------------
            # Right Click (3-finger pinch)
            if fingers_up[0] and fingers_up[1] and fingers_up[2] and dist_thumb_index<PINCH_THRESHOLD*1.2 and (now-last_right_click_time)>RIGHT_CLICK_COOLDOWN:
                pyautogui.rightClick()
                last_right_click_time=now
                gesture_text="Right Click"

            # -------------------
            # Scroll (2 fingers)
            if fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
                curr_pos = (index_tip.y*fh, index_tip.x*fw)
                if last_scroll_pos is not None:
                    dy = last_scroll_pos[0]-curr_pos[0]
                    dx = last_scroll_pos[1]-curr_pos[1]
                    if abs(dy)>abs(dx):
                        pyautogui.scroll(int(dy*SCROLL_FACTOR))
                        gesture_text="Scroll"
                    else:
                        pyautogui.hscroll(int(dx*-SCROLL_FACTOR))
                        gesture_text="H-Scroll"
                last_scroll_pos=curr_pos
                cv2.circle(frame,(int(index_tip.x*fw),int(index_tip.y*fh)),14,(0,255,255),2)
                continue
            else:
                last_scroll_pos=None

            # -------------------
            # Zoom (thumb-pinky distance)
            if dist_thumb_pinky > 0:
                if not zoom_active:
                    zoom_base = dist_thumb_pinky
                    zoom_active = True
                else:
                    delta = dist_thumb_pinky - zoom_base
                    if abs(delta) > 5:  # avoid jitter
                        pyautogui.keyDown('ctrl')
                        pyautogui.scroll(int(delta*HZOOM_FACTOR))
                        pyautogui.keyUp('ctrl')
                        zoom_base = dist_thumb_pinky
                        gesture_text = "Zoom"
            else:
                zoom_active = False

            # -------------------
            # Pinch Click & Drag
            if dist_thumb_index<PINCH_THRESHOLD:
                if not is_pinching:
                    is_pinching=True
                    pinch_start_time=now
                    gesture_text="Pinch start"
                else:
                    held=now-pinch_start_time
                    if held>DRAG_HOLD_DURATION and not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging=True
                        gesture_text="Drag"
            else:
                if is_pinching:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging=False
                        gesture_text="Drag End"
                    elif now - last_click_time < DOUBLE_CLICK_MAX_INTERVAL:
                        pyautogui.doubleClick()
                        gesture_text="Double Click"
                    else:
                        pyautogui.click()
                        gesture_text="Click"
                    last_click_time=now
                is_pinching=False

            # -------------------
            # Cursor Movement
            tx, ty = map_to_screen(index_tip.x, index_tip.y)
            tx, ty, snapped = snap_to_hotspot(tx, ty, hotspots)
            smooth_x = prev_x + (tx-prev_x)*SMOOTHING_FACTOR
            smooth_y = prev_y + (ty-prev_y)*SMOOTHING_FACTOR
            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y

            cv2.circle(frame, (int(index_tip.x*fw), int(index_tip.y*fh)), 12, (0,255,0), -1)

    if time.time()-last_input_time>3:
        is_paused=True

    if FPS_DISPLAY_ENABLED:
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(frame,f'FPS:{int(fps)}',(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

    cv2.putText(frame,gesture_text,(20,70),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.imshow("Hand Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key==27: break
    elif key==32: is_paused=not is_paused
    elif key==ord('h'):
        mx,my = pyautogui.position()
        hotspots.append((mx,my))
        config["hotspots"]=hotspots
        save_config(config)
        print(f"Saved hotspot: {mx},{my}")
    elif key==ord('r'):
        cal_data, PINCH_THRESHOLD = calibrate()
        config["calibration_data"]=cal_data
        config["pinch_threshold"]=PINCH_THRESHOLD
        save_config(config)
    elif key==ord('m'):
        is_paused=not is_paused

cap.release()
cv2.destroyAllWindows()
