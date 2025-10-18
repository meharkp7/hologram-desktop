# utils/helpers.py
import math
import time

def distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    Each point is a tuple (x, y)
    """
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def smooth(prev, target, factor):
    """
    Smooths a value for cursor movement
    prev = previous position
    target = target position
    factor = smoothing factor (0-1)
    """
    return prev + (target - prev) * factor

def cooldown(last_time, duration):
    """
    Returns True if enough time has passed since last_time
    last_time = previous action time
    duration = cooldown in seconds
    """
    return (time.time() - last_time) > duration
