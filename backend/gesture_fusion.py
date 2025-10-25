#!/usr/bin/env python3
import time
import pyautogui
from hand_tracking_backend import processing_loop

class CursorController:
    def __init__(self, smoothing=5):
        self.smoothing = smoothing
        self.prev_x = 0
        self.prev_y = 0
        self.dragging = False
        self.prev_pinch_distance = None

    def move_cursor(self, x, y):
        smoothed_x = self.prev_x + (x - self.prev_x) / self.smoothing
        smoothed_y = self.prev_y + (y - self.prev_y) / self.smoothing
        pyautogui.moveTo(smoothed_x, smoothed_y)
        if self.dragging:
            pyautogui.dragTo(smoothed_x, smoothed_y, duration=0.01, button='left')
        self.prev_x, self.prev_y = smoothed_x, smoothed_y

    def click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.click(button='right')

    def scroll(self, amount):
        pyautogui.scroll(amount)

    def start_drag(self):
        self.dragging = True
        pyautogui.mouseDown()

    def stop_drag(self):
        self.dragging = False
        pyautogui.mouseUp()

    def pinch_zoom(self, distance):
        if self.prev_pinch_distance is None:
            self.prev_pinch_distance = distance
            return
        delta = distance - self.prev_pinch_distance
        if abs(delta) > 5:  # threshold to avoid jitter
            self.scroll(int(delta))
        self.prev_pinch_distance = distance


def main():
    cursor = CursorController()
    print("Starting gesture fusion with pinch-drag and pinch-zoom... Press Ctrl+C to exit.")

    def gesture_callback(gesture_name, x, y, extra=None):
        if gesture_name == "pinch":
            cursor.click()
        elif gesture_name == "pinch_hold":
            cursor.start_drag()
        elif gesture_name == "pinch_release":
            cursor.stop_drag()
        elif gesture_name == "swipe":
            cursor.move_cursor(x, y)
        elif gesture_name == "three_finger_pinch":
            cursor.right_click()
        elif gesture_name == "two_finger_swipe_up":
            cursor.scroll(100)
        elif gesture_name == "two_finger_swipe_down":
            cursor.scroll(-100)
        elif gesture_name == "two_finger_pinch" and extra is not None:
            cursor.pinch_zoom(extra)  # extra should be the distance between fingers

    cfg = {
        "model_path": "hand_model.tflite",
        "camera_id": 0,
        "mirror": True,
        "max_num_hands": 1
    }

    try:
        processing_loop(cfg, gesture_callback)
    except KeyboardInterrupt:
        print("Exiting gesture fusion.")


if __name__ == "__main__":
    main()