# utils.py

import threading

class HandTrackingState:
    """
    Thread-safe object representing the current state of hand tracking.
    All modules read from this to get cursor/gesture info.
    """
    def __init__(self):
        self.lock = threading.Lock()
        # Cursor position (screen coords)
        self.cursor_x = 0
        self.cursor_y = 0
        # Gesture flags
        self.is_pinching = False
        self.is_dragging = False
        self.is_right_click = False
        self.is_scrolling = False
        self.scroll_dx = 0
        self.scroll_dy = 0
        self.is_zooming = False
        self.zoom_percent = 100
        # Additional info for gestures
        self.custom_gesture = None
        self.last_input_time = 0

    def update_cursor(self, x, y):
        with self.lock:
            self.cursor_x = x
            self.cursor_y = y
            self.last_input_time = __import__('time').time()

    def update_pinch(self, pinching, dragging=False):
        with self.lock:
            self.is_pinching = pinching
            self.is_dragging = dragging
            self.last_input_time = __import__('time').time()

    def update_right_click(self, flag=True):
        with self.lock:
            self.is_right_click = flag
            self.last_input_time = __import__('time').time()

    def update_scroll(self, dx, dy):
        with self.lock:
            self.is_scrolling = True
            self.scroll_dx = dx
            self.scroll_dy = dy
            self.last_input_time = __import__('time').time()

    def reset_scroll(self):
        with self.lock:
            self.is_scrolling = False
            self.scroll_dx = 0
            self.scroll_dy = 0

    def update_zoom(self, percent):
        with self.lock:
            self.is_zooming = True
            self.zoom_percent = percent
            self.last_input_time = __import__('time').time()

    def reset_zoom(self):
        with self.lock:
            self.is_zooming = False
            self.zoom_percent = 100

    def set_custom_gesture(self, gesture_name):
        with self.lock:
            self.custom_gesture = gesture_name
            self.last_input_time = __import__('time').time()

    def __repr__(self):
        with self.lock:
            return (f"<HandTrackingState cursor=({self.cursor_x:.1f},{self.cursor_y:.1f}) "
                    f"pinch={self.is_pinching} drag={self.is_dragging} "
                    f"right_click={self.is_right_click} scroll=({self.scroll_dx},{self.scroll_dy}) "
                    f"zoom={self.zoom_percent}% custom={self.custom_gesture}>")