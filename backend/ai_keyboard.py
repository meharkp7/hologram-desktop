# ai_keyboard.py
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QPropertyAnimation, pyqtProperty
from PyQt5.QtGui import QFont, QColor
from pynput.keyboard import Controller, Key

keyboard_controller = Controller()

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
            }}
            QPushButton:hover {{
                color: white;
                border: 2px solid #acd9da;
            }}
        """)

class AirKeyboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(16, 13, 40, 0.95); border-radius: 15px;")
        self.is_shift = False
        self.is_symbol = False
        self.hover_key = None
        self.typed_text = ""
        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        self.close_btn = QPushButton("❌")
        self.close_btn.setStyleSheet("font-size:18px;color:#acd9da;background:transparent;")
        self.close_btn.clicked.connect(self.hide_keyboard)
        self.main_layout.addWidget(self.close_btn, alignment=Qt.AlignRight)

        self.hover_label = QLabel("")
        self.hover_label.setFont(QFont("Jaivipurva", 20))
        self.hover_label.setStyleSheet("color:#acd9da;")
        self.main_layout.addWidget(self.hover_label, alignment=Qt.AlignCenter)

        self.keys_layout = QGridLayout()
        self.main_layout.addLayout(self.keys_layout)

        self.load_keys()
        self.resize(750, 370)

    def load_keys(self):
        self.keys_layout.setSpacing(5)
        self.keys_layout.setContentsMargins(10, 0, 10, 10)

        if self.is_symbol:
            keys = [
                list("!@#$%^&*()_+"),
                list("[]{};:'\",.<>?"),
                ["Shift", "ABC", "Backspace"],
                ["😀", "😂", "❤️", "👍", "🎉", "🔥", "Space", "Enter"]
            ]
        else:
            keys = [
                list("QWERTYUIOP"),
                list("ASDFGHJKL"),
                ["Shift"] + list("ZXCVBNM") + ["Backspace"],
                ["123", "😀", "Space", "Enter"]
            ]

        for i in reversed(range(self.keys_layout.count())):
            self.keys_layout.itemAt(i).widget().deleteLater()

        self.buttons = {}
        for row, row_keys in enumerate(keys):
            for col, key in enumerate(row_keys):
                btn = GlowButton(key)
                btn.setFont(QFont("Jaivipurva", 16))
                btn.setFixedHeight(55)
                btn.clicked.connect(lambda _, k=key: self.press_key(k))
                self.keys_layout.addWidget(btn, row, col)
                self.buttons[key] = btn

    def hover_key_event(self, key):
        if self.hover_key and self.hover_key in self.buttons:
            prev_btn = self.buttons[self.hover_key]
            anim = QPropertyAnimation(prev_btn, b"color")
            anim.setDuration(200)
            anim.setStartValue(prev_btn.color)
            anim.setEndValue(QColor("#100d28"))
            anim.start()

        self.hover_key = key
        self.hover_label.setText(key)

        if key in self.buttons:
            btn = self.buttons[key]
            anim = QPropertyAnimation(btn, b"color")
            anim.setDuration(200)
            anim.setStartValue(btn.color)
            anim.setEndValue(QColor("#7375db"))
            anim.start()

    def press_key(self, key):
        if key == "Shift":
            self.is_shift = not self.is_shift
        elif key in ["123", "ABC"]:
            self.is_symbol = not self.is_symbol
            self.load_keys()
        elif key == "Backspace":
            self.send_key_event("\b")
        elif key == "Space":
            self.send_key_event(" ")
        elif key == "Enter":
            self.send_key_event("\n")
        else:
            char = key.upper() if self.is_shift else key.lower()
            self.send_key_event(char)

    def send_key_event(self, char):
        if char == "\b":
            self.typed_text = self.typed_text[:-1]
            keyboard_controller.press(Key.backspace)
            keyboard_controller.release(Key.backspace)
        elif len(char) == 1 and char.isprintable():  
            self.typed_text += char
            keyboard_controller.press(char)
            keyboard_controller.release(char)
        else:
            self.typed_text += char
        print(f"Key pressed: {char}")


    def show_keyboard(self):
        self.show()

    def hide_keyboard(self):
        self.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    keyboard = AirKeyboard()
    keyboard.show_keyboard()
    sys.exit(app.exec_())