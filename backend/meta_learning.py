# meta_learning.py

import json
import os
import threading
import time
from collections import defaultdict
from utils import HandTrackingState

CONFIG_PATH = "meta_learning.json"

class MetaLearner:
    """
    Lightweight meta-learning system to optimize:
    - Gesture recognition accuracy
    - Typing patterns for AI keyboard
    Stores data locally and adapts weights over time.
    """
    def __init__(self, state: HandTrackingState):
        self.state = state
        self.running = True
        self.lock = threading.Lock()
        self.gesture_stats = defaultdict(lambda: {"count": 0, "success": 0})
        self.typing_stats = defaultdict(lambda: {"attempts": 0, "success": 0})
        self.load()

    # ------------------------
    # Public methods
    # ------------------------
    def record_gesture(self, gesture_name: str, success: bool):
        with self.lock:
            self.gesture_stats[gesture_name]["count"] += 1
            if success:
                self.gesture_stats[gesture_name]["success"] += 1

    def record_typing(self, char: str, success: bool):
        with self.lock:
            self.typing_stats[char]["attempts"] += 1
            if success:
                self.typing_stats[char]["success"] += 1

    def get_gesture_confidence(self, gesture_name: str) -> float:
        stats = self.gesture_stats.get(gesture_name)
        if stats and stats["count"] > 0:
            return stats["success"] / stats["count"]
        return 0.0

    def get_typing_confidence(self, char: str) -> float:
        stats = self.typing_stats.get(char)
        if stats and stats["attempts"] > 0:
            return stats["success"] / stats["attempts"]
        return 0.0

    def save(self):
        with self.lock:
            data = {
                "gesture_stats": dict(self.gesture_stats),
                "typing_stats": dict(self.typing_stats)
            }
            with open(CONFIG_PATH, 'w') as f:
                json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r') as f:
                    data = json.load(f)
                    self.gesture_stats.update(data.get("gesture_stats", {}))
                    self.typing_stats.update(data.get("typing_stats", {}))
            except Exception as e:
                print("Failed to load meta-learning data:", e)

    # ------------------------
    # Background loop to auto-save
    # ------------------------
    def run_loop(self, interval=5.0):
        while self.running:
            self.save()
            time.sleep(interval)

    def stop(self):
        self.running = False
        self.save()