from serpent.input_controller import InputController, MouseButton, KeyboardKey, character_keyboard_key_mapping
from serpent.sprite_locator import SpriteLocator
import time
import ctypes
import win32api
import scipy.interpolate
import numpy as np
keyboard_key_mapping = {KeyboardKey.KEY_ESCAPE.name: 1, KeyboardKey.KEY_F1.name: 59, KeyboardKey.KEY_F2.name: 60, KeyboardKey.KEY_F3.name: 61, KeyboardKey.KEY_F4.name: 62, KeyboardKey.KEY_F5.name: 63, KeyboardKey.KEY_F6.name: 64, KeyboardKey.KEY_F7.name: 65, KeyboardKey.KEY_F8.name: 66, KeyboardKey.KEY_F9.name: 67, KeyboardKey.KEY_F10.name: 68, KeyboardKey.KEY_F11.name: 87, KeyboardKey.KEY_F12.name: 88, KeyboardKey.KEY_PRINT_SCREEN.name: 183, KeyboardKey.KEY_SCROLL_LOCK.name: 70, KeyboardKey.KEY_PAUSE.name: 197, KeyboardKey.KEY_GRAVE.name: 41, KeyboardKey.KEY_BACKTICK.name: 41, KeyboardKey.KEY_1.name: 2, KeyboardKey.KEY_2.name: 3, KeyboardKey.KEY_3.name: 4, KeyboardKey.KEY_4.name: 5, KeyboardKey.KEY_5.name: 6, KeyboardKey.KEY_6.name: 7, KeyboardKey.KEY_7.name: 8, KeyboardKey.KEY_8.name: 9, KeyboardKey.KEY_9.name: 10, KeyboardKey.KEY_0.name: 11, KeyboardKey.KEY_MINUS.name: 12, KeyboardKey.KEY_DASH.name: 12, KeyboardKey.KEY_EQUALS.name: 13, KeyboardKey.KEY_BACKSPACE.name: 14, KeyboardKey.KEY_INSERT.name: 210 + 1024, KeyboardKey.KEY_HOME.name: 199 + 1024, KeyboardKey.KEY_PAGE_UP.name: 201 + 1024, KeyboardKey.KEY_NUMLOCK.name: 69, KeyboardKey.KEY_NUMPAD_DIVIDE.name: 181 + 1024, KeyboardKey.KEY_NUMPAD_SLASH.name: 181 + 1024, KeyboardKey.KEY_NUMPAD_MULTIPLY.name: 55, KeyboardKey.KEY_NUMPAD_STAR.name: 55, KeyboardKey.KEY_NUMPAD_SUBTRACT.name: 74, KeyboardKey.KEY_NUMPAD_DASH.name: 74, KeyboardKey.KEY_TAB.name: 15, KeyboardKey.KEY_Q.name: 16, KeyboardKey.KEY_W.name: 17, KeyboardKey.KEY_E.name: 18, KeyboardKey.KEY_R.name: 19, KeyboardKey.KEY_T.name: 20, KeyboardKey.KEY_Y.name: 21, KeyboardKey.KEY_U.name: 22, KeyboardKey.KEY_I.name: 23, KeyboardKey.KEY_O.name: 24, KeyboardKey.KEY_P.name: 25, KeyboardKey.KEY_LEFT_BRACKET.name: 26, KeyboardKey.KEY_RIGHT_BRACKET.name: 27, KeyboardKey.KEY_BACKSLASH.name: 43, KeyboardKey.KEY_DELETE.name: 211 + 1024, KeyboardKey.KEY_END.name: 207 + 1024, KeyboardKey.KEY_PAGE_DOWN.name: 209 + 1024, KeyboardKey.KEY_NUMPAD_7.name: 71, KeyboardKey.KEY_NUMPAD_8.name: 72, KeyboardKey.KEY_NUMPAD_9.name: 73, KeyboardKey.KEY_NUMPAD_ADD.name: 78, KeyboardKey.KEY_NUMPAD_PLUS.name: 78, KeyboardKey.KEY_CAPSLOCK.name: 58, KeyboardKey.KEY_A.name: 30, KeyboardKey.KEY_S.name: 31, KeyboardKey.KEY_D.name: 32, KeyboardKey.KEY_F.name: 33, KeyboardKey.KEY_G.name: 34, KeyboardKey.KEY_H.name: 35, KeyboardKey.KEY_J.name: 36, KeyboardKey.KEY_K.name: 37, KeyboardKey.KEY_L.name: 38, KeyboardKey.KEY_SEMICOLON.name: 39, KeyboardKey.KEY_APOSTROPHE.name: 40, KeyboardKey.KEY_RETURN.name: 28, KeyboardKey.KEY_ENTER.name: 28, KeyboardKey.KEY_NUMPAD_4.name: 75, KeyboardKey.KEY_NUMPAD_5.name: 76, KeyboardKey.KEY_NUMPAD_6.name: 77, KeyboardKey.KEY_LEFT_SHIFT.name: 42, KeyboardKey.KEY_Z.name: 44, KeyboardKey.KEY_X.name: 45, KeyboardKey.KEY_C.name: 46, KeyboardKey.KEY_V.name: 47, KeyboardKey.KEY_B.name: 48, KeyboardKey.KEY_N.name: 49, KeyboardKey.KEY_M.name: 50, KeyboardKey.KEY_COMMA.name: 51, KeyboardKey.KEY_PERIOD.name: 52, KeyboardKey.KEY_SLASH.name: 53, KeyboardKey.KEY_RIGHT_SHIFT.name: 54, KeyboardKey.KEY_UP.name: 200 + 1024, KeyboardKey.KEY_NUMPAD_1.name: 79, KeyboardKey.KEY_NUMPAD_2.name: 80, KeyboardKey.KEY_NUMPAD_3.name: 81, KeyboardKey.KEY_NUMPAD_RETURN.name: 156 + 1024, KeyboardKey.KEY_NUMPAD_ENTER.name: 156 + 1024, KeyboardKey.KEY_LEFT_CTRL.name: 29, KeyboardKey.KEY_LEFT_SUPER.name: 219 + 1024, KeyboardKey.KEY_LEFT_WINDOWS.name: 219 + 1024, KeyboardKey.KEY_LEFT_ALT.name: 56, KeyboardKey.KEY_SPACE.name: 57, KeyboardKey.KEY_RIGHT_ALT.name: 184 + 1024, KeyboardKey.KEY_RIGHT_SUPER.name: 220 + 1024, KeyboardKey.KEY_RIGHT_WINDOWS.name: 220 + 1024, KeyboardKey.KEY_APP_MENU.name: 221 + 1024, KeyboardKey.KEY_RIGHT_CTRL.name: 157 + 1024, KeyboardKey.KEY_LEFT.name: 203 + 1024, KeyboardKey.KEY_DOWN.name: 208 + 1024, KeyboardKey.KEY_RIGHT.name: 205 + 1024, KeyboardKey.KEY_NUMPAD_0.name: 82, KeyboardKey.KEY_NUMPAD_DECIMAL.name: 83, KeyboardKey.KEY_NUMPAD_PERIOD.name: 83}
mouse_button_down_mapping = {MouseButton.LEFT.name: 2, MouseButton.MIDDLE.name: 32, MouseButton.RIGHT.name: 8}
mouse_button_up_mapping = {MouseButton.LEFT.name: 4, MouseButton.MIDDLE.name: 64, MouseButton.RIGHT.name: 16}
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [('wVk', ctypes.c_ushort), ('wScan', ctypes.c_ushort), ('dwFlags', ctypes.c_ulong), ('time', ctypes.c_ulong), ('dwExtraInfo', PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [('uMsg', ctypes.c_ulong), ('wParamL', ctypes.c_short), ('wParamH', ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [('dx', ctypes.c_long), ('dy', ctypes.c_long), ('mouseData', ctypes.c_ulong), ('dwFlags', ctypes.c_ulong), ('time', ctypes.c_ulong), ('dwExtraInfo', PUL)]

class Input_I(ctypes.Union):
    _fields_ = [('ki', KeyBdInput), ('mi', MouseInput), ('hi', HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [('type', ctypes.c_ulong), ('ii', Input_I)]

class NativeWin32InputController(InputController):

    def __init__(self, game=None, **kwargs):
        if False:
            return 10
        self.game = game
        self.previous_key_collection_set = set()
        self.sprite_locator = SpriteLocator()

    def handle_keys(self, key_collection, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        key_collection_set = set(key_collection)
        keys_to_press = key_collection_set - self.previous_key_collection_set
        keys_to_release = self.previous_key_collection_set - key_collection_set
        for key in keys_to_press:
            self.press_key(key, **kwargs)
        for key in keys_to_release:
            self.release_key(key, **kwargs)
        self.previous_key_collection_set = key_collection_set

    def tap_keys(self, keys, duration=0.05, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            for key in keys:
                self.press_key(key, **kwargs)
            time.sleep(duration)
            for key in keys:
                self.release_key(key, **kwargs)

    def tap_key(self, key, duration=0.05, **kwargs):
        if False:
            while True:
                i = 10
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            self.press_key(key, **kwargs)
            time.sleep(duration)
            self.release_key(key, **kwargs)

    def press_keys(self, keys, **kwargs):
        if False:
            i = 10
            return i + 15
        for key in keys:
            self.press_key(key, **kwargs)

    def press_key(self, key, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            if keyboard_key_mapping[key.name] >= 1024:
                key = keyboard_key_mapping[key.name] - 1024
                flags = 8 | 1
            else:
                key = keyboard_key_mapping[key.name]
                flags = 8
            ii_.ki = KeyBdInput(0, key, flags, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_keys(self, keys, **kwargs):
        if False:
            while True:
                i = 10
        for key in keys:
            self.release_key(key, **kwargs)

    def release_key(self, key, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            if keyboard_key_mapping[key.name] >= 1024:
                key = keyboard_key_mapping[key.name] - 1024
                flags = 8 | 1 | 2
            else:
                key = keyboard_key_mapping[key.name]
                flags = 8 | 2
            ii_.ki = KeyBdInput(0, key, flags, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def type_string(self, string, duration=0.05, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            for character in string:
                keys = character_keyboard_key_mapping.get(character)
                if keys is not None:
                    self.tap_keys(keys, duration=duration, **kwargs)

    def move(self, x=None, y=None, duration=0.25, absolute=True, interpolate=True, **kwargs):
        if False:
            return 10
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            if absolute:
                x += self.game.window_geometry['x_offset']
                y += self.game.window_geometry['y_offset']
                current_pixel_coordinates = win32api.GetCursorPos()
                start_coordinates = self._to_windows_coordinates(*current_pixel_coordinates)
                end_coordinates = self._to_windows_coordinates(x, y)
                if interpolate:
                    coordinates = self._interpolate_mouse_movement(start_windows_coordinates=start_coordinates, end_windows_coordinates=end_coordinates)
                else:
                    coordinates = [end_coordinates]
                for (x, y) in coordinates:
                    extra = ctypes.c_ulong(0)
                    ii_ = Input_I()
                    ii_.mi = MouseInput(x, y, 0, 1 | 32768, 0, ctypes.pointer(extra))
                    x = Input(ctypes.c_ulong(0), ii_)
                    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
                    time.sleep(duration / len(coordinates))
            else:
                x = int(x)
                y = int(y)
                coordinates = self._interpolate_mouse_movement(start_windows_coordinates=(0, 0), end_windows_coordinates=(x, y))
                for (x, y) in coordinates:
                    extra = ctypes.c_ulong(0)
                    ii_ = Input_I()
                    ii_.mi = MouseInput(x, y, 0, 1, 0, ctypes.pointer(extra))
                    x = Input(ctypes.c_ulong(0), ii_)
                    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
                    time.sleep(duration / len(coordinates))

    def click_down(self, button=MouseButton.LEFT, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(0, 0, 0, mouse_button_down_mapping[button.name], 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def click_up(self, button=MouseButton.LEFT, **kwargs):
        if False:
            return 10
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(0, 0, 0, mouse_button_up_mapping[button.name], 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def click(self, button=MouseButton.LEFT, duration=0.05, **kwargs):
        if False:
            print('Hello World!')
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            self.click_down(button=button, **kwargs)
            time.sleep(duration)
            self.click_up(button=button, **kwargs)

    def click_screen_region(self, button=MouseButton.LEFT, screen_region=None, **kwargs):
        if False:
            return 10
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            screen_region_coordinates = self.game.screen_regions.get(screen_region)
            x = (screen_region_coordinates[1] + screen_region_coordinates[3]) // 2
            y = (screen_region_coordinates[0] + screen_region_coordinates[2]) // 2
            self.move(x, y)
            self.click(button=button, **kwargs)

    def click_sprite(self, button=MouseButton.LEFT, sprite=None, game_frame=None, **kwargs):
        if False:
            print('Hello World!')
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            sprite_location = self.sprite_locator.locate(sprite=sprite, game_frame=game_frame)
            if sprite_location is None:
                return False
            x = (sprite_location[1] + sprite_location[3]) // 2
            y = (sprite_location[0] + sprite_location[2]) // 2
            self.move(x, y)
            self.click(button=button, **kwargs)
            return True

    def click_string(self, query_string, button=MouseButton.LEFT, game_frame=None, fuzziness=2, ocr_preset=None, **kwargs):
        if False:
            while True:
                i = 10
        import serpent.ocr
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            string_location = serpent.ocr.locate_string(query_string, game_frame.frame, fuzziness=fuzziness, ocr_preset=ocr_preset, offset_x=game_frame.offset_x, offset_y=game_frame.offset_y)
            if string_location is not None:
                x = (string_location[1] + string_location[3]) // 2
                y = (string_location[0] + string_location[2]) // 2
                self.move(x, y)
                self.click(button=button, **kwargs)
                return True
            return False

    def drag(self, button=MouseButton.LEFT, x0=None, y0=None, x1=None, y1=None, duration=0.25, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            self.move(x=x0, y=y0)
            self.click_down(button=button)
            self.move(x=x1, y=y1, duration=duration)
            self.click_up(button=button)

    def drag_screen_region_to_screen_region(self, button=MouseButton.LEFT, start_screen_region=None, end_screen_region=None, duration=1, **kwargs):
        if False:
            print('Hello World!')
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            start_screen_region_coordinates = self._extract_screen_region_coordinates(start_screen_region)
            end_screen_region_coordinates = self._extract_screen_region_coordinates(end_screen_region)
            self.drag(button=button, x0=start_screen_region_coordinates[0], y0=start_screen_region_coordinates[1], x1=end_screen_region_coordinates[0], y1=end_screen_region_coordinates[1], duration=duration, **kwargs)

    def scroll(self, clicks=1, direction='DOWN', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'force' in kwargs and kwargs['force'] is True or self.game_is_focused:
            clicks = clicks * (1 if direction == 'UP' else -1) * 120
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(0, 0, clicks, 2048, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    @staticmethod
    def _to_windows_coordinates(x=0, y=0):
        if False:
            for i in range(10):
                print('nop')
        display_width = win32api.GetSystemMetrics(0)
        display_height = win32api.GetSystemMetrics(1)
        windows_x = x * 65535 // display_width
        windows_y = y * 65535 // display_height
        return (windows_x, windows_y)

    @staticmethod
    def _interpolate_mouse_movement(start_windows_coordinates, end_windows_coordinates, steps=20):
        if False:
            while True:
                i = 10
        x_coordinates = [start_windows_coordinates[0], end_windows_coordinates[0]]
        y_coordinates = [start_windows_coordinates[1], end_windows_coordinates[1]]
        if x_coordinates[0] == x_coordinates[1]:
            x_coordinates[1] += 1
        if y_coordinates[0] == y_coordinates[1]:
            y_coordinates[1] += 1
        interpolation_func = scipy.interpolate.interp1d(x_coordinates, y_coordinates)
        intermediate_x_coordinates = np.linspace(start_windows_coordinates[0], end_windows_coordinates[0], steps + 1)[1:]
        coordinates = list(map(lambda x: (int(round(x)), int(interpolation_func(x))), intermediate_x_coordinates))
        return coordinates