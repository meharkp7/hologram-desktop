import json
import os
from typing import Dict

class GestureLibrary:
    def __init__(self, config_path="gesture_config.json"):
        self.config_path = config_path
        self.gesture_map = self.load_gesture_map()

    def load_gesture_map(self) -> Dict[str, Dict]:
        """Load gesture→action mappings from JSON (creates default if missing)."""
        if not os.path.exists(self.config_path):
            default_map = {
                "PINCH_START": {"action": "LEFT_CLICK", "description": "Select / Click"},
                "PINCH_HOLD": {"action": "DRAG", "description": "Hold to drag"},
                "PINCH_RELEASE": {"action": "RELEASE", "description": "Release drag"},
                "THREE_FINGER_PINCH": {"action": "RIGHT_CLICK", "description": "Context menu"},
                "TWO_FINGER_SWIPE_UP": {"action": "SCROLL_UP", "description": "Scroll page up"},
                "TWO_FINGER_SWIPE_DOWN": {"action": "SCROLL_DOWN", "description": "Scroll page down"},
                "ZOOM_IN": {"action": "ZOOM_IN", "description": "Magnify"},
                "ZOOM_OUT": {"action": "ZOOM_OUT", "description": "Shrink"},
                "PALM_LEFT": {"action": "NEXT_APP", "description": "Switch application"},
                "PALM_RIGHT": {"action": "PREV_APP", "description": "Switch back"},
                "DRAW_L": {"action": "LOCK_SCREEN", "description": "Lock workstation"}
            }
            with open(self.config_path, "w") as f:
                json.dump(default_map, f, indent=4)
            return default_map
        with open(self.config_path, "r") as f:
            return json.load(f)

    def interpret(self, gesture: str) -> Dict:
        """Return corresponding action info for a detected gesture."""
        return self.gesture_map.get(gesture, {"action": None, "description": "Unknown gesture"})

    def update_gesture(self, gesture: str, action: str, description: str = ""):
        """Update mapping for a gesture and save to file."""
        self.gesture_map[gesture] = {"action": action, "description": description}
        with open(self.config_path, "w") as f:
            json.dump(self.gesture_map, f, indent=4)
        return True