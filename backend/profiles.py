# profiles.py

from utils import HandTrackingState

class ProfileManager:
    """
    Simplified manager: no profiles, everything is always enabled.
    """
    def __init__(self, state: HandTrackingState):
        self.state = state

        # Enable all gestures and AI keyboard by default
        self.state.ai_keyboard_enabled = True
        self.state.active_gestures = [
            "pinch_click", "scroll", "drag",
            "next_slide", "prev_slide", "laser_pointer",
            "play_pause", "volume", "skip"
        ]

    # No profile switching needed
    def apply_current_mode(self, hand_state: HandTrackingState):
        pass