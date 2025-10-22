# overlays.py

import threading
import time
import json
from utils import HandTrackingState

CONFIG_PATH = "overlays.json"

class OverlayManager:
    """
    Manages AR-like overlays:
    - Sticky notes
    - Timers
    - Floating widgets
    """
    def __init__(self, state: HandTrackingState):
        self.state = state
        self.overlays = []  # list of dicts: {"type": str, "pos": (x,y), "content": str, "duration": float}
        self.lock = threading.Lock()
        self.load_overlays()
        self._running = True
        self._thread = threading.Thread(target=self._overlay_loop, daemon=True)
        self._thread.start()

    # ------------------------
    # Overlay management
    # ------------------------
    def add_overlay(self, overlay_type: str, pos: tuple, content: str, duration: float = 0):
        """
        duration=0 means persistent until manually removed
        """
        with self.lock:
            self.overlays.append({"type": overlay_type, "pos": pos, "content": content, "duration": duration, "start_time": time.time()})

    def remove_overlay(self, index: int):
        with self.lock:
            if 0 <= index < len(self.overlays):
                del self.overlays[index]

    def clear_overlays(self):
        with self.lock:
            self.overlays.clear()

    def get_active_overlays(self):
        """
        Return overlays that should currently be visible (duration not expired)
        """
        now = time.time()
        with self.lock:
            active = []
            for ov in self.overlays:
                if ov["duration"] == 0 or (now - ov["start_time"]) < ov["duration"]:
                    active.append(ov)
            return active

    # ------------------------
    # Persistence
    # ------------------------
    def load_overlays(self):
        try:
            with open(CONFIG_PATH, 'r') as f:
                self.overlays = json.load(f).get("overlays", [])
        except Exception:
            self.overlays = []

    def save_overlays(self):
        with self.lock:
            data = {"overlays": self.overlays}
            with open(CONFIG_PATH, 'w') as f:
                json.dump(data, f, indent=2)

    # ------------------------
    # Internal loop
    # ------------------------
    def _overlay_loop(self):
        """
        Can be expanded to handle visual rendering on screen using OpenCV
        or integration with hand-tracking cursor
        """
        while self._running:
            active = self.get_active_overlays()
            # Currently just updates state with overlay info for frontend
            self.state.active_overlays = active
            time.sleep(0.05)

    def stop(self):
        self._running = False
        self._thread.join()
        self.save_overlays()