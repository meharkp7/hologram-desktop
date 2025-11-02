# ai_keyboard.py
import sys
import os
import difflib
import threading
import urllib.request
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QGridLayout, QVBoxLayout, QLabel,
    QHBoxLayout, QSizePolicy, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QPropertyAnimation, pyqtProperty, QTimer
from PyQt5.QtGui import QFont, QColor
from pynput.keyboard import Controller, Key

keyboard_controller = Controller()

WORDS_FILENAME = "words_alpha.txt"
WORDS_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
MIN_SUGGESTIONS = 3
MAX_SUGGESTIONS = 5
SUGGESTION_CUTOFF = 0.4

# emoji map: keywords -> emoji list (extend as needed)
EMOJI_MAP = {
    "happy": ["😊", "😁", "😄"],
    "love": ["❤️", "😍", "💖"],
    "lol": ["😂", "🤣"],
    "ok": ["👌", "👍"],
    "fire": ["🔥"],
    "party": ["🎉", "🥳"],
    "thumb": ["👍"],
    "sad": ["😢", "😞"],
    "heart": ["❤️", "💘", "💓"],
    "cool": ["😎", "🆒"],
}

# small fallback word list (used only if large list not available)
FALLBACK_WORDS = [
    "hello","help","happy","how","are","you","yes","no","thanks","thank","please",
    "good","great","fine","love","like","happy","awesome","amazing","fun","cool",
    "keyboard","mouse","click","scroll","zoom","drag","pinch","space","enter",
    "python","code","project","test","example","suggest","complete","typing",
    "autocomplete","predict","model","local","offline","online","emoji","smile"
]

def download_wordlist_if_needed(filename=WORDS_FILENAME, url=WORDS_URL, force=False):
    p = Path(filename)
    if p.exists() and not force:
        return True
    try:
        # Download in a separate thread to avoid blocking UI on first run
        def dl():
            try:
                urllib.request.urlretrieve(url, filename)
            except Exception:
                pass
        t = threading.Thread(target=dl, daemon=True)
        t.start()
        t.join(timeout=12)  # try for a few seconds; if it doesn't complete, fallback will be used
        return p.exists()
    except Exception:
        return False

def load_wordlist(filename=WORDS_FILENAME):
    if download_wordlist_if_needed(filename):
        try:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                words = [w.strip().lower() for w in f if w.strip()]
                return words
        except Exception:
            pass
    return FALLBACK_WORDS[:]

class GlowButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self._bg_color = QColor("#100d28")
        self.update_style()
    def getColor(self):
        return self._bg_color
    def setColor(self, color):
        self._bg_color = color
        self.update_style()
    color = pyqtProperty(QColor, getColor, setColor)
    def update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._bg_color.name()};
                color: #acd9da;
                border-radius: 12px;
                border: 2px solid #00003b;
                padding: 6px;
            }}
            QPushButton:hover {{
                color: white;
                border: 2px solid #acd9da;
            }}
        """)

class AirKeyboard(QWidget):
    def __init__(self, preload_words=True):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(16, 13, 40, 0.95); border-radius: 15px;")
        self.is_shift = False
        self.is_symbol = False
        self.is_number = False
        self.is_emoji = False
        self.hover_key = None
        self.typed_text = ""
        self.wordlist = []
        if preload_words:
            # load heavy list in background; use fallback instantly
            self.wordlist = load_wordlist()  # immediate attempt
            # if it's fallback and file missing, try background download for next run
            if self.wordlist == FALLBACK_WORDS:
                threading.Thread(target=download_wordlist_if_needed, daemon=True).start()
        self.init_ui()
        self.suggestion_timer = QTimer()
        self.suggestion_timer.timeout.connect(self.update_suggestions)
        self.suggestion_timer.start(220)

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        # top row: close button and typed-text preview
        top_row = QHBoxLayout()
        self.close_btn = QPushButton("❌")
        self.close_btn.setStyleSheet("font-size:18px;color:#acd9da;background:transparent;")
        self.close_btn.clicked.connect(self.hide_keyboard)
        top_row.addWidget(self.close_btn, alignment=Qt.AlignLeft)

        self.preview_label = QLabel("")
        self.preview_label.setFont(QFont("Jaivipurva", 16))
        self.preview_label.setStyleSheet("color:#acd9da;")
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_row.addWidget(self.preview_label, alignment=Qt.AlignCenter)

        self.main_layout.addLayout(top_row)

        # suggestion bar (scrollable)
        self.suggestion_area = QScrollArea()
        self.suggestion_area.setFixedHeight(60)
        self.suggestion_area.setFrameShape(QFrame.NoFrame)
        self.suggestion_area.setWidgetResizable(True)
        sug_container = QWidget()
        self.suggestion_layout = QHBoxLayout(sug_container)
        self.suggestion_layout.setContentsMargins(6,6,6,6)
        self.suggestion_layout.setSpacing(8)
        self.suggestion_area.setWidget(sug_container)
        self.main_layout.addWidget(self.suggestion_area)

        # keys layout area
        self.keys_layout = QGridLayout()
        self.main_layout.addLayout(self.keys_layout)

        self.load_keys()
        self.resize(820, 420)

    def load_keys(self):
        # define layout sets
        letters_keys = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            ["Shift"] + list("ZXCVBNM") + ["Backspace"],
            ["123", "😀", "Space", "Enter"]
        ]
        numbers_keys = [
            list("1234567890"),
            list("-=/+*()[]"),
            ["ABC", "Backspace"],
            ["😀", "Space", "Enter"]
        ]
        symbols_keys = [
            list("!@#$%^&*()_+"),
            list("[]{};:'\",.<>?"),
            ["ABC", "Backspace"],
            ["😀", "Space", "Enter"]
        ]
        emoji_keys = [
            ["😀","😂","❤️","👍","🎉","🔥","😢","😎","🥳","💖"],
            ["😡","😱","😴","🤔","🤩","🙏","💯","🎶","☀️","🌙"],
            ["ABC","Backspace","Space","Enter"]
        ]
        if getattr(self, "is_emoji", False):
            keys = emoji_keys
        elif self.is_symbol:
            keys = symbols_keys
        elif self.is_number:
            keys = numbers_keys
        else:
            keys = letters_keys


        # clear existing widgets
        for i in reversed(range(self.keys_layout.count())):
            item = self.keys_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()

        self.buttons = {}
        for row, row_keys in enumerate(keys):
            for col, key in enumerate(row_keys):
                btn = GlowButton(key)
                btn.setFont(QFont("Jaivipurva", 16))
                btn.setFixedHeight(56)
                btn.clicked.connect(lambda _, k=key: self.press_key(k))
                self.keys_layout.addWidget(btn, row, col)
                self.buttons[key] = btn

    def hover_key_event(self, key):
        # update hover visual state
        if self.hover_key and self.hover_key in self.buttons:
            prev_btn = self.buttons[self.hover_key]
            anim = QPropertyAnimation(prev_btn, b"color")
            anim.setDuration(140)
            anim.setStartValue(prev_btn.color)
            anim.setEndValue(QColor("#100d28"))
            anim.start()
        self.hover_key = key
        self.preview_label.setText(self.typed_text[-48:])
        if key in self.buttons:
            btn = self.buttons[key]
            anim = QPropertyAnimation(btn, b"color")
            anim.setDuration(140)
            anim.setStartValue(btn.color)
            anim.setEndValue(QColor("#7375db"))
            anim.start()

    def press_key(self, key):
        # button click handler (also used by fusion overlay)
        if key == "Shift":
            self.is_shift = not self.is_shift
            self.load_keys()
            return
        if key == "123":
            self.is_number = True
            self.is_symbol = False
            self.load_keys()
            return
        if key in ["!@#$%^&*()_+", "[]{};:'\",.<>?"]:
            self.is_symbol = True
            self.is_number = False
            self.load_keys()
            return
        if key == "Backspace":
            self.send_key_event("\b")
            return
        if key == "Space":
            self.send_key_event(" ")
            return
        if key == "Enter":
            self.send_key_event("\n")
            return
        if key == "😀": 
            self.is_emoji = True
            self.is_symbol = False
            self.is_number = False
            self.load_keys()
            return
        elif key == "ABC":
            self.is_emoji = False
            self.is_symbol = False
            self.is_number = False
            self.load_keys()
            return

        # normal character (letters, numbers, emoji)
        char = key.upper() if (self.is_shift and len(key)==1 and key.isalpha()) else key
        self.send_key_event(char)

    def send_key_event(self, char):
        """
        Updates internal typed_text preview and sends actual keystroke to the OS via pynput.
        For emojis and multi-char glyphs we only update preview (pynput may not support all glyphs).
        """
        # Backspace
        if char == "\b":
            if self.typed_text:
                self.typed_text = self.typed_text[:-1]
                try:
                    keyboard_controller.press(Key.backspace)
                    keyboard_controller.release(Key.backspace)
                except Exception:
                    pass
        elif char == "\n":
            self.typed_text += "\n"
            try:
                keyboard_controller.press(Key.enter)
                keyboard_controller.release(Key.enter)
            except Exception:
                pass
        elif len(char) == 1 and ord(char) < 128:
            # ASCII printable - send via pynput
            self.typed_text += char
            try:
                keyboard_controller.press(char)
                keyboard_controller.release(char)
            except Exception:
                pass
        else:
            # emoji / multi-byte / symbol: update preview but don't always send via pynput
            self.typed_text += char
            # attempt to send if keyboard controller can handle (best-effort)
            try:
                keyboard_controller.press(char)
                keyboard_controller.release(char)
            except Exception:
                pass

        # refresh preview and suggestions
        self.preview_label.setText(self.typed_text[-48:])
        self.update_suggestions()

    def update_suggestions(self):
        # rebuild suggestion bar based on last incomplete word
        for i in reversed(range(self.suggestion_layout.count())):
            item = self.suggestion_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()

        last_word = ""
        if self.typed_text and self.typed_text.strip():
            parts = self.typed_text.rstrip().split()
            if parts:
                last_word = parts[-1].lower()
        if not last_word:
            return

        # emoji suggestions (keyword match or prefix)
        emoji_suggestions = []
        for k, ems in EMOJI_MAP.items():
            if last_word == k or last_word.startswith(k) or k.startswith(last_word):
                emoji_suggestions.extend(ems)
        # word suggestions from huge list
        words = []
        try:
            if self.wordlist:
                words = difflib.get_close_matches(last_word, self.wordlist, n=MAX_SUGGESTIONS, cutoff=SUGGESTION_CUTOFF)
        except Exception:
            words = []

        # merge emoji suggestions first, then words; limit total displayed
        merged = []
        for e in emoji_suggestions:
            if e not in merged:
                merged.append(e)
        for w in words:
            if len(merged) >= MAX_SUGGESTIONS:
                break
            if w not in merged:
                merged.append(w)

        # fallback: if nothing, present few prefix matches (cheap)
        if not merged:
            prefix_matches = []
            if self.wordlist:
                try:
                    for w in self.wordlist:
                        if w.startswith(last_word) and w != last_word:
                            prefix_matches.append(w)
                            if len(prefix_matches) >= MAX_SUGGESTIONS:
                                break
                except Exception:
                    pass
            merged = prefix_matches[:MAX_SUGGESTIONS]

        # create buttons for suggestions
        for sugg in merged[:MAX_SUGGESTIONS]:
            btn = GlowButton(sugg)
            btn.setFont(QFont("Jaivipurva", 14))
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda _, s=sugg: self.apply_suggestion(s))
            self.suggestion_layout.addWidget(btn)

    def apply_suggestion(self, suggestion):
        # replace last incomplete word with suggestion, then append a space
        words = self.typed_text.rstrip().split()
        if words:
            last = words[-1]
            # compute suffix to type after current last
            words[-1] = suggestion
            new_text = " ".join(words) + " "
            suffix = new_text[len(self.typed_text):]
        else:
            new_text = suggestion + " "
            suffix = new_text
        # send suffix characters to OS (best-effort)
        for ch in suffix:
            if ch == " ":
                try:
                    keyboard_controller.press(Key.space)
                    keyboard_controller.release(Key.space)
                except Exception:
                    pass
            elif ch == "\n":
                try:
                    keyboard_controller.press(Key.enter)
                    keyboard_controller.release(Key.enter)
                except Exception:
                    pass
            else:
                try:
                    keyboard_controller.press(ch)
                    keyboard_controller.release(ch)
                except Exception:
                    pass
        self.typed_text = new_text
        self.preview_label.setText(self.typed_text[-48:])
        self.update_suggestions()

    def show_keyboard(self):
        self.show()

    def hide_keyboard(self):
        self.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    kb = AirKeyboard(preload_words=True)
    kb.show_keyboard()
    sys.exit(app.exec_())