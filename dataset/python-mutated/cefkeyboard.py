"""
Cef Keyboard Manager.
Cef Keyboard management is complex, so we outsourced it to this file for
better readability.
"""
from kivy.core.window import Window
from .cefpython import cefpython

class CEFKeyboardManagerSingleton:
    is_shift1 = False
    is_shift2 = False
    is_ctrl1 = False
    is_ctrl2 = False
    is_alt1 = False
    is_alt2 = False

    def __init__(self, *largs, **dargs):
        if False:
            i = 10
            return i + 15
        pass

    def reset_all_modifiers(self):
        if False:
            while True:
                i = 10
        self.is_shift1 = False
        self.is_shift2 = False
        self.is_ctrl1 = False
        self.is_ctrl2 = False
        self.is_alt1 = False
        self.is_alt2 = False

    def kivy_keyboard_on_textinput(self, browser, window, text):
        if False:
            for i in range(10):
                print('nop')
        ' Kivy ~ > 1.9.2 with SDL2 window, uses on_textinput instead of\n        on_key_down\n        '
        modifiers = list()
        keycode = (ord(text), text)
        self.process_key_down(browser, None, keycode, text, modifiers)

    def kivy_on_key_down(self, browser, keyboard, keycode, text, modifiers):
        if False:
            print('Hello World!')
        whitelist = (9, 8, 13, 27)
        if Window.__class__.__module__ == 'kivy.core.window.window_sdl2' and keycode[0] not in whitelist:
            return
        self.process_key_down(browser, keyboard, keycode, text, modifiers)

    def process_key_down(self, browser, keyboard, key, text, modifiers):
        if False:
            for i in range(10):
                print('nop')
        if key[1] == 'special':
            return
        if key[0] == 27:
            browser.GetFocusedFrame().ExecuteJavascript('__kivy__on_escape()')
            return
        if key[0] == 13:
            text = '\r'
        if key[0] in (306, 305):
            modifiers.append('ctrl')
        cef_modifiers = cefpython.EVENTFLAG_NONE
        if 'shift' in modifiers:
            cef_modifiers |= cefpython.EVENTFLAG_SHIFT_DOWN
        if 'ctrl' in modifiers:
            cef_modifiers |= cefpython.EVENTFLAG_CONTROL_DOWN
        if 'alt' in modifiers:
            cef_modifiers |= cefpython.EVENTFLAG_ALT_DOWN
        if 'capslock' in modifiers:
            cef_modifiers |= cefpython.EVENTFLAG_CAPS_LOCK_ON
        keycode = self.get_windows_key_code(key[0])
        charcode = key[0]
        if text:
            charcode = ord(text)
        if key[0] not in range(35, 40 + 1):
            key_event = {'type': cefpython.KEYEVENT_RAWKEYDOWN, 'windows_key_code': keycode, 'character': charcode, 'unmodified_character': charcode, 'modifiers': cef_modifiers}
            browser.SendKeyEvent(key_event)
        if text:
            key_event = {'type': cefpython.KEYEVENT_CHAR, 'windows_key_code': keycode, 'character': charcode, 'unmodified_character': charcode, 'modifiers': cef_modifiers}
            browser.SendKeyEvent(key_event)
        if key[0] == 304:
            self.is_shift1 = True
        elif key[0] == 303:
            self.is_shift2 = True
        elif key[0] == 306:
            self.is_ctrl1 = True
        elif key[0] == 305:
            self.is_ctrl2 = True
        elif key[0] == 308:
            self.is_alt1 = True
        elif key[0] == 313:
            self.is_alt2 = True

    def kivy_on_key_up(self, browser, keyboard, key):
        if False:
            for i in range(10):
                print('nop')
        if key[0] == -1:
            return
        cef_modifiers = cefpython.EVENTFLAG_NONE
        if self.is_shift1 or self.is_shift2:
            cef_modifiers |= cefpython.EVENTFLAG_SHIFT_DOWN
        if self.is_ctrl1 or self.is_ctrl2:
            cef_modifiers |= cefpython.EVENTFLAG_CONTROL_DOWN
        if self.is_alt1:
            cef_modifiers |= cefpython.EVENTFLAG_ALT_DOWN
        keycode = self.get_windows_key_code(key[0])
        charcode = key[0]
        key_event = {'type': cefpython.KEYEVENT_KEYUP, 'windows_key_code': keycode, 'character': charcode, 'unmodified_character': charcode, 'modifiers': cef_modifiers}
        browser.SendKeyEvent(key_event)
        if key[0] == 304:
            self.is_shift1 = False
        elif key[0] == 303:
            self.is_shift2 = False
        elif key[0] == 306:
            self.is_ctrl1 = False
        elif key[0] == 305:
            self.is_ctrl2 = False
        elif key[0] == 308:
            self.is_alt1 = False
        elif key[0] == 313:
            self.is_alt2 = False

    def get_windows_key_code(self, kivycode):
        if False:
            i = 10
            return i + 15
        cefcode = kivycode
        other_keys_map = {'27': 27, '282': 112, '283': 113, '284': 114, '285': 115, '286': 116, '287': 117, '288': 118, '289': 119, '290': 120, '291': 121, '292': 122, '293': 123, '9': 9, '304': 16, '303': 16, '306': 17, '305': 17, '308': 18, '313': 225, '8': 8, '13': 13, '316': 42, '302': 145, '19': 19, '277': 45, '127': 46, '278': 36, '279': 35, '280': 33, '281': 34, '276': 37, '273': 38, '275': 39, '274': 40, '96': 192, '45': 189, '61': 187, '91': 219, '93': 221, '92': 220, '311': 91, '59': 186, '39': 222, '44': 188, '46': 190, '47': 91, '319': 0}
        if str(kivycode) in other_keys_map:
            cefcode = other_keys_map[str(kivycode)]
        return cefcode
CEFKeyboardManager = CEFKeyboardManagerSingleton()