# ai_keyboard.py

from utils import HandTrackingState
from pynput.keyboard import Controller as KeyboardController, Key
import threading
import time

keyboard = KeyboardController()

class AIKeyboard:
    def __init__(self, state: HandTrackingState):
        self.state = state
        self.buffer = ""
        self.predictions = []
        self.lock = threading.Lock()
        self.running = True

    # ------------------------
    # Public methods
    # ------------------------
    def type_char(self, char: str):
        with self.lock:
            self.buffer += char
            keyboard.press(char)
            keyboard.release(char)
            self._update_predictions()

    def backspace(self):
        with self.lock:
            if self.buffer:
                self.buffer = self.buffer[:-1]
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
                self._update_predictions()

    def commit_prediction(self, prediction: str):
        """Replace buffer with prediction"""
        with self.lock:
            # remove old buffer
            for _ in range(len(self.buffer)):
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
            self.buffer = prediction
            for c in prediction:
                keyboard.press(c)
                keyboard.release(c)
            self._update_predictions()

    def reset(self):
        with self.lock:
            self.buffer = ""
            self.predictions = []

    # ------------------------
    # Predictive typing (simple demo, can plug ML model later)
    # ------------------------
    def _update_predictions(self):
        """Simple predictive model: suggest words starting with last typed letter"""
        common_words = ["hello", "help", "home", "world", "python", "predict", "gesture", "keyboard"]
        if self.buffer:
            self.predictions = [w for w in common_words if w.startswith(self.buffer)]
        else:
            self.predictions = []

    def get_predictions(self):
        with self.lock:
            return list(self.predictions)

    # ------------------------
    # Voice-Gesture fusion (placeholder for integration)
    # ------------------------
    def type_via_gesture(self, char: str):
        """Simulate typing via a recognized gesture"""
        self.type_char(char)

    def type_via_speech(self, word: str):
        """Simulate typing a word from speech"""
        self.commit_prediction(word)

    # ------------------------
    # Optional continuous thread
    # ------------------------
    def run_loop(self, interval=0.05):
        while self.running:
            # Example: could read state here and auto-type gestures
            if self.state.custom_gesture:
                self.type_via_gesture(self.state.custom_gesture[0])
                self.state.custom_gesture = None
            time.sleep(interval)

    def stop(self):
        self.running = False