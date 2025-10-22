#!/usr/bin/env python3
"""
main_backend.py - Entry point for touchless desktop control backend.

Orchestrates:
- hand tracking (hand_tracking_backend.py)
- gesture handling (gesture_library.py)
- AI keyboard (ai_keyboard.py)
- AI assistant (ai_assistant.py)
- overlays (overlays.py)
- user profiles/modes (profiles.py)
- optional meta-learning (meta_learning.py)
- configuration management (config_manager.py)
"""

import threading
import time
from hand_tracking_backend import start_hand_tracking, stop_hand_tracking
from gesture_library import GestureLibrary
from ai_keyboard import AIKeyboard
from ai_assistant import AIAssistant
from overlays import OverlayManager
from profiles import ProfileManager
from meta_learning import MetaLearner
from config_manager import ConfigManager
from utils import HandTrackingState

# -----------------------
# Global Managers
# -----------------------
config = ConfigManager(path="config.json")
profiles = ProfileManager(config)
gestures = GestureLibrary(config)
keyboard_ai = AIKeyboard(config)
assistant_ai = AIAssistant(config)
overlays = OverlayManager(config)
meta_learning = MetaLearner(config)

# -----------------------
# Hand Tracking State
# -----------------------
hand_state = HandTrackingState()

# -----------------------
# Thread Handles
# -----------------------
_ht_thread = None
_gesture_thread = None

# -----------------------
# Worker Functions
# -----------------------
def _hand_tracking_worker():
    start_hand_tracking(hand_state)

def _gesture_worker():
    while True:
        gestures.process(hand_state)
        profiles.apply_current_mode(hand_state)
        keyboard_ai.process(hand_state)
        assistant_ai.process(hand_state)
        overlays.update(hand_state)
        meta_learning.adapt(hand_state)
        time.sleep(0.005)

# -----------------------
# Public API
# -----------------------
def launch_backend():
    """Start backend threads safely."""
    global _ht_thread, _gesture_thread
    if _ht_thread is None or not _ht_thread.is_alive():
        _ht_thread = threading.Thread(target=_hand_tracking_worker, daemon=True)
        _ht_thread.start()
    if _gesture_thread is None or not _gesture_thread.is_alive():
        _gesture_thread = threading.Thread(target=_gesture_worker, daemon=True)
        _gesture_thread.start()
    return _ht_thread, _gesture_thread

def shutdown_backend():
    """Stop backend threads cleanly."""
    stop_hand_tracking()
    global _ht_thread, _gesture_thread
    if _ht_thread:
        _ht_thread.join()
        _ht_thread = None
    if _gesture_thread:
        _gesture_thread.join(timeout=0.1)
        _gesture_thread = None

# -----------------------
# Main Entry
# -----------------------
def main():
    print("Starting Touchless Desktop Backend...")
    config.load()
    profiles.switch_profile("Work")

    launch_backend()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down backend...")
    finally:
        shutdown_backend()
        print("Backend shutdown complete.")

if __name__ == "__main__":
    main()