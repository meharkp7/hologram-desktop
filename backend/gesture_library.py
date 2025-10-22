# gesture_library.py

from utils import HandTrackingState
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
import time

mouse = MouseController()
keyboard = KeyboardController()

class GestureLibrary:
    def __init__(self, state: HandTrackingState):
        self.state = state
        self.last_right_click_time = 0
        self.right_click_cooldown = 0.5  # seconds
        self.double_click_interval = 0.4
        self.last_click_time = 0

    def process(self):
        """Call this every frame to execute gestures."""
        now = time.time()

        # PINCH / DRAG
        if self.state.is_pinching:
            if self.state.is_dragging:
                try:
                    mouse.press(Button.left)
                except Exception:
                    pass
            else:
                if now - self.last_click_time <= self.double_click_interval:
                    mouse.click(Button.left, 2)
                else:
                    mouse.click(Button.left, 1)
                self.last_click_time = now
        else:
            try:
                mouse.release(Button.left)
            except Exception:
                pass

        # RIGHT CLICK
        if self.state.is_right_click and (now - self.last_right_click_time) > self.right_click_cooldown:
            mouse.click(Button.right, 1)
            self.last_right_click_time = now

        # SCROLL
        if self.state.is_scrolling:
            mouse.scroll(self.state.scroll_dx, self.state.scroll_dy)
            self.state.reset_scroll()

        # ZOOM
        if self.state.is_zooming:
            keyboard.press(Key.ctrl)
            mouse.scroll(0, int((self.state.zoom_percent - 100) / 10))
            keyboard.release(Key.ctrl)
            self.state.reset_zoom()

        # CUSTOM GESTURES
        if self.state.custom_gesture:
            # Example mapping
            if self.state.custom_gesture.lower() == "l_shape":
                keyboard.press(Key.cmd)
                keyboard.press("l")
                keyboard.release("l")
                keyboard.release(Key.cmd)
            # reset after executing
            self.state.custom_gesture = None