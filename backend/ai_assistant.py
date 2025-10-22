# ai_assistant.py

from utils import HandTrackingState
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button
import threading
import time
import subprocess

keyboard = KeyboardController()
mouse = MouseController()

class AIAssistant:
    def __init__(self, state: HandTrackingState):
        self.state = state
        self.running = True
        self.lock = threading.Lock()
        # basic command mapping
        self.commands = {
            "open_chrome": self.open_chrome,
            "open_terminal": self.open_terminal,
            "close_window": self.close_window,
            "switch_app": self.switch_app,
            "summarize": self.summarize_page_placeholder
        }

    # ------------------------
    # Public methods
    # ------------------------
    def execute_command(self, command_name: str):
        with self.lock:
            func = self.commands.get(command_name)
            if func:
                func()
            else:
                print(f"Unknown command: {command_name}")

    # ------------------------
    # Basic desktop actions
    # ------------------------
    def open_chrome(self):
        try:
            if subprocess.os.name == 'posix':
                subprocess.Popen(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"])
            else:
                subprocess.Popen(["chrome"])
            print("Chrome opened")
        except Exception as e:
            print("Failed to open Chrome:", e)

    def open_terminal(self):
        try:
            if subprocess.os.name == 'posix':
                subprocess.Popen(["/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal"])
            else:
                subprocess.Popen(["cmd.exe"])
            print("Terminal opened")
        except Exception as e:
            print("Failed to open Terminal:", e)

    def close_window(self):
        keyboard.press(Key.alt)
        keyboard.press(Key.f4)
        keyboard.release(Key.f4)
        keyboard.release(Key.alt)
        print("Close window triggered")

    def switch_app(self):
        keyboard.press(Key.alt)
        keyboard.press(Key.tab)
        keyboard.release(Key.tab)
        keyboard.release(Key.alt)
        print("Switch app triggered")

    def summarize_page_placeholder(self):
        # Placeholder for local LLM summarization
        print("Summarize command received (LLM integration pending)")

    # ------------------------
    # Gesture/voice integration loop
    # ------------------------
    def run_loop(self, interval=0.05):
        while self.running:
            if self.state.custom_gesture:
                cmd = self.map_gesture_to_command(self.state.custom_gesture)
                if cmd:
                    self.execute_command(cmd)
                self.state.custom_gesture = None
            time.sleep(interval)

    def stop(self):
        self.running = False

    # ------------------------
    # Gesture → Command mapping
    # ------------------------
    def map_gesture_to_command(self, gesture_name: str):
        mapping = {
            "L_shape": "switch_app",
            "V_shape": "open_chrome",
            "fist": "close_window",
            "point": "summarize"
        }
        return mapping.get(gesture_name)