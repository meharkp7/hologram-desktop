# backend/gesture_processor.py
import math
import time
from backend import actions

# -----------------------
# Gesture State Variables
# -----------------------
is_pinching = False
is_dragging = False
is_selecting = False
drag_start_time = 0
drag_hold_duration = 0.25
last_pinch_time = 0
click_cooldown = 0.5
right_click_cooldown = 0.5
pinch_confirmation_delay = 0.05  # 50ms delay for pinch
snap_radius = 50  # pixels radius to snap to button centers

# -----------------------
# Core Gesture Functions
# -----------------------
def distance(pt1, pt2):
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def snap_to_targets(x, y, targets):
    """Snap cursor to nearest target if within snap_radius"""
    for tx, ty in targets:
        if distance((x, y), (tx, ty)) < snap_radius:
            return tx, ty
    return x, y

def process_gesture(landmarks, frame_w, frame_h, snap_targets=[]):
    global is_pinching, is_dragging, is_selecting, drag_start_time, last_pinch_time

    # Extract finger tips
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Finger bases
    index_base = landmarks[5]
    middle_base = landmarks[9]
    ring_base = landmarks[13]
    pinky_base = landmarks[17]

    # Determine finger states
    is_index_up = index_tip.y < index_base.y
    is_middle_up = middle_tip.y < middle_base.y
    is_ring_up = ring_tip.y < ring_base.y
    is_pinky_up = pinky_tip.y < pinky_base.y

    is_fist = not is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up
    is_open_palm = is_index_up and is_middle_up and is_ring_up and is_pinky_up

    # Convert normalized coordinates to pixels
    index_pos = (index_tip.x * frame_w, index_tip.y * frame_h)
    middle_pos = (middle_tip.x * frame_w, middle_tip.y * frame_h)
    thumb_pos = (thumb_tip.x * frame_w, thumb_tip.y * frame_h)

    # -------------------
    # Left Click (Pinch) with Confirmation Delay
    # -------------------
    dist_pinch = distance(thumb_pos, index_pos)
    if dist_pinch < 30 and (time.time() - last_pinch_time) > click_cooldown and not is_pinching:
        time.sleep(pinch_confirmation_delay)
        actions.left_click()
        is_pinching = True
        last_pinch_time = time.time()
    elif dist_pinch >= 30:
        is_pinching = False

    # -------------------
    # Right Click (Three-Finger Pinch)
    # -------------------
    dist_index_middle = distance(index_pos, middle_pos)
    if dist_pinch < 30 and dist_index_middle < 30 and (time.time() - last_pinch_time) > right_click_cooldown:
        actions.right_click()
        last_pinch_time = time.time()

    # -------------------
    # Scrolling
    # -------------------
    if is_index_up and is_middle_up and not is_ring_up and not is_pinky_up:
        scroll_amount = middle_tip.y - index_tip.y
        actions.scroll_vertical(scroll_amount * 100)  # scale for sensitivity

    # -------------------
    # Dragging / Selection
    # -------------------
    if is_fist:
        if not is_selecting:
            actions.click_and_hold()
            is_selecting = True
            drag_start_time = time.time()
        if (time.time() - drag_start_time) > drag_hold_duration and not is_dragging:
            is_dragging = True

    if is_open_palm and is_dragging:
        actions.release_click()
        is_selecting = False
        is_dragging = False
        drag_start_time = 0

    if not is_fist and is_selecting:
        actions.release_click()
        is_selecting = False

    # -------------------
    # Cursor Control + Snapping
    # -------------------
    x, y = index_pos
    x, y = snap_to_targets(x, y, snap_targets)

    return int(x), int(y), is_dragging