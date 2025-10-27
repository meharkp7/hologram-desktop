class AirKeyboard:
    def __init__(self, key_map=None, predictive_dict=None):
        self.key_map = key_map or self._default_key_map()
        self.predictive_dict = predictive_dict or {}
        self.typed_text = ""

    def detect_hover(self, hand_position):
        return self._map_finger_to_key(hand_position)

    def detect_tap(self, hand_position, pinch_gesture):
        key = self._map_finger_to_key(hand_position)
        if key is not None and pinch_gesture:
            self.key_pressed(key)
            return key
        return None

    def key_pressed(self, key):
        self.typed_text += key
        self._update_predictive_dict(key)

    def predict_next(self):
        words = self.typed_text.split()
        if not words:
            return []
        last_word = words[-1]
        suggestions = [w for w in self.predictive_dict.keys() if w.startswith(last_word)]
        return suggestions[:3]

    def swipe_typing(self, swipe_path):
        return "".join([self._map_finger_to_key(pos) or "" for pos in swipe_path])

    # ---- Helper Methods ----
    def _map_finger_to_key(self, finger_pos):
        for key, pos in self.key_map.items():
            if self._is_inside(finger_pos, pos):
                return key
        return None

    def _update_predictive_dict(self, key):
        self.predictive_dict[key] = self.predictive_dict.get(key, 0) + 1

    def _is_inside(self, finger_pos, key_pos):
        x, y = finger_pos
        kx, ky, w, h = key_pos
        return kx <= x <= kx + w and ky <= y <= ky + h

    def _default_key_map(self):
        keys = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {k: (i * 50, 0, 50, 50) for i, k in enumerate(keys)}