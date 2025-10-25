# utils.py
import math
from enum import Enum

# ------------------------
# ENUMS & STATES
# ------------------------
class HandTrackingState(Enum):
    NO_HAND = 0
    HAND_DETECTED = 1
    PINCH = 2
    DRAG = 3
    RIGHT_CLICK = 4
    SCROLL = 5
    ZOOM = 6

# ------------------------
# MATH / UTILITY FUNCTIONS
# ------------------------
def distance(p1, p2):
    """Euclidean distance between two points (x, y)"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

def clamp(val, min_val, max_val):
    """Clamp a value between min_val and max_val"""
    return max(min_val, min(max_val, val))

def exponential_smooth(prev, new, alpha=0.35):
    """Simple exponential smoothing"""
    return prev * (1 - alpha) + new * alpha

def lerp(a, b, t):
    """Linear interpolation between a and b"""
    return a + (b - a) * t

def normalize(value, min_val, max_val):
    """Normalize value to 0-1 range"""
    return clamp((value - min_val) / (max_val - min_val), 0.0, 1.0)

def denormalize(value, min_val, max_val):
    """Map normalized value 0-1 back to range [min_val, max_val]"""
    return clamp(value * (max_val - min_val) + min_val, min_val, max_val)

def average(lst):
    """Return mean of list"""
    return sum(lst) / len(lst) if lst else 0.0

def midpoint(p1, p2):
    """Return midpoint between two points"""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

def is_hand_resting(prev_point, current_point, threshold=5):
    if not prev_point:
        return False
    dx = abs(current_point[0] - prev_point[0])
    dy = abs(current_point[1] - prev_point[1])
    return math.hypot(dx, dy) < threshold
