import json
import time
from threading import Thread
from hand_tracking_backend import HandTracker
from ai_keyboard import get_hovered_key, press_key

class GestureSystem:
    def __init__(self, gestures_file="gestures.json", poll_interval=0.05):
        self.poll_interval = poll_interval
        self.gesture_map = self.load_gestures(gestures_file)
        self.hand_tracker = HandTracker()
        self.active_gesture = None
        self.running = False
        self.last_trigger_time = {}

    def load_gestures(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        # Verify all required IDs (3,4,5) exist
        for gid in [3,4,5]:
            if str(gid) not in data:
                raise ValueError(f"Gesture ID {gid} missing in gestures.json")
        return data

    def get_active_gesture(self):
        return self.active_gesture

    def gesture_triggered(self, gesture_id):
        now = time.time()
        last_time = self.last_trigger_time.get(gesture_id, 0)
        cooldown = self.gesture_map[str(gesture_id)].get("cooldown", 0.3)
        if now - last_time >= cooldown:
            self.last_trigger_time[gesture_id] = now
            return True
        return False

    def handle_gesture(self, gesture_id):
        gesture_info = self.gesture_map.get(str(gesture_id), None)
        if not gesture_info:
            return
        if self.gesture_triggered(gesture_id):
            action = gesture_info.get("action")
            if action == "press_key":
                key = gesture_info.get("key")
                hovered = get_hovered_key()
                if hovered:
                    press_key(key or hovered)
            elif action == "custom":
                func = gesture_info.get("function")
                if func:
                    func()

    def poll_gestures(self):
        while self.running:
            hands = self.hand_tracker.get_hands()
            for hand in hands:
                gesture_id = hand.get("gesture_id")
                if gesture_id:
                    self.active_gesture = gesture_id
                    self.handle_gesture(gesture_id)
            time.sleep(self.poll_interval)

    def start(self):
        self.running = True
        t = Thread(target=self.poll_gestures, daemon=True)
        t.start()

    def stop(self):
        self.running = False

# Example usage:
if __name__ == "__main__":
    gs = GestureSystem()
    gs.start()
    try:
        while True:
            active = gs.get_active_gesture()
            if active:
                print(f"Active Gesture: {active}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        gs.stop()