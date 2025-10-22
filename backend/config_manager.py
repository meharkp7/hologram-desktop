# config_manager.py

import json
import os
from threading import Lock

class ConfigManager:
    """
    Load / Save / Update configs for:
    - Calibration
    - Hotspots
    - Profiles
    - Overlays
    """
    def __init__(self, path: str):
        self.path = path
        self.lock = Lock()
        self.config = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self.config = json.load(f)
            except Exception:
                self.config = {}
        else:
            self.config = {}

    def save(self):
        with self.lock:
            with open(self.path, 'w') as f:
                json.dump(self.config, f, indent=2)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        with self.lock:
            self.config[key] = value
            self.save()

    def update_nested(self, key, subkey, value):
        with self.lock:
            if key not in self.config:
                self.config[key] = {}
            self.config[key][subkey] = value
            self.save()
            