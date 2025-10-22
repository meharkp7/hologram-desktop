# profiles.py

import json
import os
from threading import Lock
from utils import HandTrackingState

CONFIG_PATH = "profiles.json"

class ProfileManager:
    """
    Manages multiple user profiles / modes:
    - Work Mode: typing + app switching
    - Presentation Mode: slide control + laser-pointer
    - Media Mode: play/pause/volume/skip
    """
    def __init__(self, state: HandTrackingState):
        self.state = state
        self.lock = Lock()
        self.profiles = {}
        self.current_profile = None
        self.load_profiles()

    # ------------------------
    # Public methods
    # ------------------------
    def load_profiles(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r') as f:
                    data = json.load(f)
                    self.profiles = data.get("profiles", {})
                    self.current_profile = data.get("current_profile")
            except Exception as e:
                print("Failed to load profiles:", e)
        else:
            # Default profiles
            self.profiles = {
                "Work": {"gestures": ["pinch_click", "scroll", "drag"], "ai_keyboard": True},
                "Presentation": {"gestures": ["next_slide", "prev_slide", "laser_pointer"], "ai_keyboard": False},
                "Media": {"gestures": ["play_pause", "volume", "skip"], "ai_keyboard": False}
            }
            self.current_profile = "Work"
            self.save_profiles()

    def save_profiles(self):
        with self.lock:
            data = {"profiles": self.profiles, "current_profile": self.current_profile}
            with open(CONFIG_PATH, 'w') as f:
                json.dump(data, f, indent=2)

    def switch_profile(self, profile_name: str):
        if profile_name in self.profiles:
            with self.lock:
                self.current_profile = profile_name
                self.apply_profile()
                self.save_profiles()
        else:
            print(f"Profile '{profile_name}' does not exist.")

    def apply_profile(self):
        """
        Applies the current profile settings to HandTrackingState.
        For example, enabling/disabling AI keyboard, certain gestures, etc.
        """
        profile = self.profiles.get(self.current_profile)
        if not profile:
            return
        gestures = profile.get("gestures", [])
        ai_keyboard = profile.get("ai_keyboard", False)

        # Example: toggle AI keyboard state
        self.state.ai_keyboard_enabled = ai_keyboard
        self.state.active_gestures = gestures

    def add_profile(self, profile_name: str, gestures: list, ai_keyboard: bool):
        with self.lock:
            self.profiles[profile_name] = {"gestures": gestures, "ai_keyboard": ai_keyboard}
            self.save_profiles()

    def remove_profile(self, profile_name: str):
        with self.lock:
            if profile_name in self.profiles:
                del self.profiles[profile_name]
                if self.current_profile == profile_name:
                    self.current_profile = next(iter(self.profiles), None)
                self.save_profiles()