#!/usr/bin/env python3
"""
main_backend.py - Touchless desktop backend (simplified, DEBUG).

- Hand tracking
- Gesture processing
- AI keyboard & assistant
- Overlay updates
- Meta-learning
- No separate profiles: all gestures always enabled
"""

import threading
import time
from hand_tracking_backend import start_hand_tracking, stop_hand_tracking
from gesture_library import GestureLibrary
from ai_keyboard import AIKeyboard
from ai_assistant import AIAssistant
from overlays import OverlayManager
from profiles import ProfileManager  # simplified version
from meta_learning import MetaLearner
from config_manager import ConfigManager
from utils import HandTrackingState

# -----------------------
# Global Managers
# -----------------------
config = ConfigManager(path="config.json")
hand_state = HandTrackingState()
profiles = ProfileManager(hand_state)
gestures = GestureLibrary(config)
keyboard_ai = AIKeyboard(config)
assistant_ai = AIAssistant(config)
overlays = OverlayManager(config)
meta_learning = MetaLearner(config)

# -----------------------
# Thread Handles
# -----------------------
_ht_thread = None
_gesture_thread = None
_shutdown_flag = threading.Event()

# -----------------------
# Worker Functions
# -----------------------
def _hand_tracking_worker():
    print("[HandTracking] Worker started")
    try:
        start_hand_tracking(hand_state)
    except Exception as e:
        print("[HandTracking] ERROR:", e)
    print("[HandTracking] Worker exited")

def _gesture_worker():
    print("[Gesture] Worker started")
    while not _shutdown_flag.is_set():
        try:
            gestures.process(hand_state)
            profiles.apply_current_mode(hand_state)  # does nothing now
            keyboard_ai.process(hand_state)
            assistant_ai.process(hand_state)
            overlays.update(hand_state)
            meta_learning.adapt(hand_state)
        except Exception as e:
            print("[Gesture] ERROR:", e)
        time.sleep(0.005)
    print("[Gesture] Worker exited")

# -----------------------
# Public API
# -----------------------
def launch_backend():
    """Start backend threads safely."""
    global _ht_thread, _gesture_thread
    print("[Backend] Launching threads...")
    if _ht_thread is None or not _ht_thread.is_alive():
        _ht_thread = threading.Thread(target=_hand_tracking_worker, daemon=True)
        _ht_thread.start()
        print("[Backend] Hand tracking thread started")
    if _gesture_thread is None or not _gesture_thread.is_alive():
        _gesture_thread = threading.Thread(target=_gesture_worker, daemon=True)
        _gesture_thread.start()
        print("[Backend] Gesture thread started")
    return _ht_thread, _gesture_thread

def shutdown_backend():
    """Stop backend threads cleanly."""
    print("[Backend] Shutting down threads...")
    _shutdown_flag.set()
    stop_hand_tracking()
    global _ht_thread, _gesture_thread
    if _ht_thread:
        _ht_thread.join(timeout=2)
        _ht_thread = None
        print("[Backend] Hand tracking thread stopped")
    if _gesture_thread:
        _gesture_thread.join(timeout=2)
        _gesture_thread = None
        print("[Backend] Gesture thread stopped")

# -----------------------
# Main Entry
# -----------------------
def main():
    print("[Main] Starting Touchless Desktop Backend...")
    config.load()
    launch_backend()
    print("[Main] Backend launched successfully")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt received. Shutting down...")
    finally:
        shutdown_backend()
        print("[Main] Backend shutdown complete.")

if __name__ == "__main__":
    main()