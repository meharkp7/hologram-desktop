import cv2
import mediapipe as mp
import pyautogui
import time
import math
import statistics

DEFAULT_PINCH_THRESHOLD = 30

def calibrate(cap, hands):
    points = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    cal_data = {}
    pinch_samples = []
    frame_w, frame_h = int(cap.get(3)), int(cap.get(4))

    for point in points:
        print(f"Calibrating: place your index on {point} and pinch to capture.")
        captured = False
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            cv2.putText(frame, f"Pinch on {point}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                index_tip = lm.landmark[8]
                thumb_tip = lm.landmark[4]
                dist = math.hypot((thumb_tip.x - index_tip.x) * frame_w,
                                  (thumb_tip.y - index_tip.y) * frame_h)
                if dist < DEFAULT_PINCH_THRESHOLD:
                    cal_data[point] = (index_tip.x, index_tip.y)
                    pinch_samples.append(dist)
                    captured = True
                    time.sleep(0.5)
                    break

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                exit()

        if not captured:
            print(f"Failed to capture {point}, exiting.")
            exit()

    pinch_threshold = int(statistics.mean(pinch_samples) + statistics.stdev(pinch_samples) if len(pinch_samples) > 1 else DEFAULT_PINCH_THRESHOLD)
    print("Calibration complete.")
    return cal_data, pinch_threshold