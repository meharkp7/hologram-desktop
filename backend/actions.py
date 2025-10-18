# backend/actions.py
import pyautogui
import time

# -----------------------
# Configuration / Tuning
# -----------------------
SMOOTHING_FACTOR = 0.5
SCROLL_FACTOR = 0.5
CLICK_DELAY = 0.05  

# -----------------------
# Cursor Actions
# -----------------------
prev_x, prev_y = None, None

def move_cursor(x, y, smooth=True):
    """Move cursor to (x, y). Optionally smooth movement."""
    global prev_x, prev_y
    if smooth:
        if prev_x is None or prev_y is None:
            prev_x, prev_y = x, y
        new_x = prev_x + (x - prev_x) * SMOOTHING_FACTOR
        new_y = prev_y + (y - prev_y) * SMOOTHING_FACTOR
        pyautogui.moveTo(new_x, new_y)
        prev_x, prev_y = new_x, new_y
    else:
        pyautogui.moveTo(x, y)
        prev_x, prev_y = x, y

# -----------------------
# Click Actions
# -----------------------
def left_click():
    pyautogui.click()

def double_left_click():
    pyautogui.doubleClick()

def right_click():
    pyautogui.rightClick()

def double_right_click():
    right_click()
    time.sleep(CLICK_DELAY)
    right_click()

def click_and_hold():
    pyautogui.mouseDown(button='left')

def release_click():
    pyautogui.mouseUp(button='left')

# -----------------------
# Drag Actions
# -----------------------
def drag_to(x, y, smooth=True):
    """Drag mouse to position while holding click."""
    move_cursor(x, y, smooth)

# -----------------------
# Scroll Actions
# -----------------------
def scroll_vertical(amount):
    """Scroll vertically. Positive = up, Negative = down."""
    pyautogui.scroll(int(amount * SCROLL_FACTOR))

def scroll_horizontal(amount):
    """Scroll horizontally. Positive = right, Negative = left."""
    pyautogui.hscroll(int(amount * SCROLL_FACTOR))

def reset_cursor_smoothing():
    global prev_x, prev_y
    prev_x, prev_y = None, None