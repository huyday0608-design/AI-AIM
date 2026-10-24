import platform
import threading

from logic.logger import logger

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


class InputBackend:
    def __init__(self):
        self.mode = "none"
        self.available = False
        self.error = None
        self._pressed = set()
        self._lock = threading.Lock()
        self._mouse_controller = None
        self._mouse_button = None

        if IS_WINDOWS:
            self._init_win32()
        else:
            self._init_pynput()

    def _init_win32(self):
        try:
            import win32api
            import win32con
            import ctypes
            from ctypes import wintypes

            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", wintypes.LONG),
                    ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_size_t),
                ]

            class _INPUT_UNION(ctypes.Union):
                _fields_ = [("mi", MOUSEINPUT)]

            class INPUT(ctypes.Structure):
                _anonymous_ = ("_input",)
                _fields_ = [
                    ("type", wintypes.DWORD),
                    ("_input", _INPUT_UNION),
                ]

            self.win32api = win32api
            self.win32con = win32con
            self._ctypes = ctypes
            self._INPUT = INPUT
            self._SendInput = ctypes.windll.user32.SendInput
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception:
                pass
            self.mode = "win32"
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            logger.warning(f"[Input] Win32 input backend is unavailable: {exc}")

    def _init_pynput(self):
        try:
            from pynput import keyboard, mouse

            self.keyboard = keyboard
            self.mouse = mouse
            self._mouse_controller = mouse.Controller()
            self._mouse_button = mouse.Button
            self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
            self._mouse_listener = mouse.Listener(on_click=self._on_click)
            self._keyboard_listener.daemon = True
            self._mouse_listener.daemon = True
            self._keyboard_listener.start()
            self._mouse_listener.start()
            self.mode = "pynput"
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            logger.warning(f"[Input] pynput backend is unavailable: {exc}")

    def _on_key_press(self, key):
        name = self._normalize_key(key)
        if name:
            with self._lock:
                self._pressed.add(name)

    def _on_key_release(self, key):
        name = self._normalize_key(key)
        if name:
            with self._lock:
                self._pressed.discard(name)

    def _on_click(self, x, y, button, pressed):
        name = self._normalize_mouse_button(button)
        if not name:
            return
        with self._lock:
            if pressed:
                self._pressed.add(name)
            else:
                self._pressed.discard(name)

    def is_pressed(self, key_name):
        if not key_name or key_name == "None":
            return False

        if self.mode == "win32":
            from logic.buttons import Buttons

            key_code = Buttons.KEY_CODES.get(key_name)
            return bool(key_code and self.win32api.GetAsyncKeyState(key_code) < 0)

        with self._lock:
            return key_name in self._pressed

    def move_mouse(self, x, y):
        if self.mode == "win32":
            int_x = int(x)
            int_y = int(y)
            if hasattr(self, "_SendInput"):
                try:
                    inp = self._INPUT()
                    inp.type = 0  # INPUT_MOUSE
                    inp.mi.dx = int_x
                    inp.mi.dy = int_y
                    inp.mi.dwFlags = 0x0001  # MOUSEEVENTF_MOVE
                    inp.mi.time = 0
                    inp.mi.dwExtraInfo = 0
                    if self._SendInput(1, self._ctypes.byref(inp), self._ctypes.sizeof(self._INPUT)) == 1:
                        return True
                except Exception:
                    pass
            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_MOVE, int_x, int_y, 0, 0)
            return True
        if self._mouse_controller is not None:
            self._mouse_controller.move(int(x), int(y))
            return True
        return False

    def left_down(self):
        if self.mode == "win32":
            if hasattr(self, "_SendInput"):
                try:
                    inp = self._INPUT()
                    inp.type = 0
                    inp.mi.dwFlags = 0x0002  # MOUSEEVENTF_LEFTDOWN
                    if self._SendInput(1, self._ctypes.byref(inp), self._ctypes.sizeof(self._INPUT)) == 1:
                        return True
                except Exception:
                    pass
            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            return True
        if self._mouse_controller is not None and self._mouse_button is not None:
            self._mouse_controller.press(self._mouse_button.left)
            return True
        return False

    def left_up(self):
        if self.mode == "win32":
            if hasattr(self, "_SendInput"):
                try:
                    inp = self._INPUT()
                    inp.type = 0
                    inp.mi.dwFlags = 0x0004  # MOUSEEVENTF_LEFTUP
                    if self._SendInput(1, self._ctypes.byref(inp), self._ctypes.sizeof(self._INPUT)) == 1:
                        return True
                except Exception:
                    pass
            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return True
        if self._mouse_controller is not None and self._mouse_button is not None:
            self._mouse_controller.release(self._mouse_button.left)
            return True
        return False

    def _normalize_mouse_button(self, button):
        if not hasattr(button, "name"):
            return None
        return {
            "left": "LeftMouseButton",
            "right": "RightMouseButton",
            "middle": "MiddleMouseButton",
        }.get(button.name)

    def _normalize_key(self, key):
        char = getattr(key, "char", None)
        if char:
            if char.isalpha():
                return char.upper()
            if char.isdigit():
                return f"Key{char}"
            if char == " ":
                return "Space"

        name = getattr(key, "name", None)
        if not name:
            return None

        if name.startswith("f") and name[1:].isdigit():
            return name.upper()

        return {
            "esc": "Escape",
            "space": "Space",
            "enter": "Enter",
            "tab": "Tab",
            "backspace": "Backspace",
            "delete": "Delete",
            "insert": "Ins",
            "home": "Home",
            "end": "End",
            "page_up": "PageUp",
            "page_down": "PageDown",
            "up": "UpArrow",
            "down": "DownArrow",
            "left": "LeftArrow",
            "right": "RightArrow",
            "shift": "LeftShift",
            "shift_l": "LeftShift",
            "shift_r": "RightShift",
            "ctrl": "LeftControl",
            "ctrl_l": "LeftControl",
            "ctrl_r": "RightControl",
            "alt": "LeftAlt",
            "alt_l": "LeftAlt",
            "alt_r": "RightAlt",
            "caps_lock": "CapsLock",
            "num_lock": "NumLock",
            "scroll_lock": "ScrollLock",
        }.get(name)


input_backend = InputBackend()
