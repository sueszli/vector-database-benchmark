import unittest
import pygame.constants
K_AND_KSCAN_COMMON_NAMES = ('UNKNOWN', 'BACKSPACE', 'TAB', 'CLEAR', 'RETURN', 'PAUSE', 'ESCAPE', 'SPACE', 'COMMA', 'MINUS', 'PERIOD', 'SLASH', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'SEMICOLON', 'EQUALS', 'LEFTBRACKET', 'BACKSLASH', 'RIGHTBRACKET', 'DELETE', 'KP0', 'KP1', 'KP2', 'KP3', 'KP4', 'KP5', 'KP6', 'KP7', 'KP8', 'KP9', 'KP_PERIOD', 'KP_DIVIDE', 'KP_MULTIPLY', 'KP_MINUS', 'KP_PLUS', 'KP_ENTER', 'KP_EQUALS', 'UP', 'DOWN', 'RIGHT', 'LEFT', 'INSERT', 'HOME', 'END', 'PAGEUP', 'PAGEDOWN', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'NUMLOCK', 'CAPSLOCK', 'SCROLLOCK', 'RSHIFT', 'LSHIFT', 'RCTRL', 'LCTRL', 'RALT', 'LALT', 'RMETA', 'LMETA', 'LSUPER', 'RSUPER', 'MODE', 'HELP', 'PRINT', 'SYSREQ', 'BREAK', 'MENU', 'POWER', 'EURO', 'KP_0', 'KP_1', 'KP_2', 'KP_3', 'KP_4', 'KP_5', 'KP_6', 'KP_7', 'KP_8', 'KP_9', 'NUMLOCKCLEAR', 'SCROLLLOCK', 'RGUI', 'LGUI', 'PRINTSCREEN', 'CURRENCYUNIT', 'CURRENCYSUBUNIT')
K_AND_KSCAN_COMMON_OVERLAPS = (('KP0', 'KP_0'), ('KP1', 'KP_1'), ('KP2', 'KP_2'), ('KP3', 'KP_3'), ('KP4', 'KP_4'), ('KP5', 'KP_5'), ('KP6', 'KP_6'), ('KP7', 'KP_7'), ('KP8', 'KP_8'), ('KP9', 'KP_9'), ('NUMLOCK', 'NUMLOCKCLEAR'), ('SCROLLOCK', 'SCROLLLOCK'), ('LSUPER', 'LMETA', 'LGUI'), ('RSUPER', 'RMETA', 'RGUI'), ('PRINT', 'PRINTSCREEN'), ('BREAK', 'PAUSE'), ('EURO', 'CURRENCYUNIT'))

def create_overlap_set(constant_names):
    if False:
        return 10
    'Helper function to find overlapping constant values/names.\n\n    Returns a set of fronzensets:\n        set(frozenset(names of overlapping constants), ...)\n    '
    overlap_dict = {}
    for name in constant_names:
        value = getattr(pygame.constants, name)
        overlap_dict.setdefault(value, set()).add(name)
    overlaps = set()
    for overlap_names in overlap_dict.values():
        if len(overlap_names) > 1:
            overlaps.add(frozenset(overlap_names))
    return overlaps

class KConstantsTests(unittest.TestCase):
    """Test K_* (key) constants."""
    K_SPECIFIC_NAMES = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'QUOTE', 'BACKQUOTE', 'EXCLAIM', 'QUOTEDBL', 'HASH', 'DOLLAR', 'AMPERSAND', 'LEFTPAREN', 'RIGHTPAREN', 'ASTERISK', 'PLUS', 'COLON', 'LESS', 'GREATER', 'QUESTION', 'AT', 'CARET', 'UNDERSCORE', 'PERCENT')
    K_NAMES = tuple(('K_' + n for n in K_AND_KSCAN_COMMON_NAMES + K_SPECIFIC_NAMES))

    def test_k__existence(self):
        if False:
            i = 10
            return i + 15
        'Ensures K constants exist.'
        for name in self.K_NAMES:
            self.assertTrue(hasattr(pygame.constants, name), f'missing constant {name}')

    def test_k__type(self):
        if False:
            while True:
                i = 10
        'Ensures K constants are the correct type.'
        for name in self.K_NAMES:
            value = getattr(pygame.constants, name)
            self.assertIs(type(value), int)

    def test_k__value_overlap(self):
        if False:
            print('Hello World!')
        'Ensures no unexpected K constant values overlap.'
        EXPECTED_OVERLAPS = {frozenset(('K_' + n for n in item)) for item in K_AND_KSCAN_COMMON_OVERLAPS}
        overlaps = create_overlap_set(self.K_NAMES)
        self.assertSetEqual(overlaps, EXPECTED_OVERLAPS)

class KscanConstantsTests(unittest.TestCase):
    """Test KSCAN_* (scancode) constants."""
    KSCAN_SPECIFIC_NAMES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'APOSTROPHE', 'GRAVE', 'INTERNATIONAL1', 'INTERNATIONAL2', 'INTERNATIONAL3', 'INTERNATIONAL4', 'INTERNATIONAL5', 'INTERNATIONAL6', 'INTERNATIONAL7', 'INTERNATIONAL8', 'INTERNATIONAL9', 'LANG1', 'LANG2', 'LANG3', 'LANG4', 'LANG5', 'LANG6', 'LANG7', 'LANG8', 'LANG9', 'NONUSBACKSLASH', 'NONUSHASH')
    KSCAN_NAMES = tuple(('KSCAN_' + n for n in K_AND_KSCAN_COMMON_NAMES + KSCAN_SPECIFIC_NAMES))

    def test_kscan__existence(self):
        if False:
            while True:
                i = 10
        'Ensures KSCAN constants exist.'
        for name in self.KSCAN_NAMES:
            self.assertTrue(hasattr(pygame.constants, name), f'missing constant {name}')

    def test_kscan__type(self):
        if False:
            print('Hello World!')
        'Ensures KSCAN constants are the correct type.'
        for name in self.KSCAN_NAMES:
            value = getattr(pygame.constants, name)
            self.assertIs(type(value), int)

    def test_kscan__value_overlap(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures no unexpected KSCAN constant values overlap.'
        EXPECTED_OVERLAPS = {frozenset(('KSCAN_' + n for n in item)) for item in K_AND_KSCAN_COMMON_OVERLAPS}
        overlaps = create_overlap_set(self.KSCAN_NAMES)
        self.assertSetEqual(overlaps, EXPECTED_OVERLAPS)

class KmodConstantsTests(unittest.TestCase):
    """Test KMOD_* (key modifier) constants."""
    KMOD_CONSTANTS = ('KMOD_NONE', 'KMOD_LSHIFT', 'KMOD_RSHIFT', 'KMOD_SHIFT', 'KMOD_LCTRL', 'KMOD_RCTRL', 'KMOD_CTRL', 'KMOD_LALT', 'KMOD_RALT', 'KMOD_ALT', 'KMOD_LMETA', 'KMOD_RMETA', 'KMOD_META', 'KMOD_NUM', 'KMOD_CAPS', 'KMOD_MODE', 'KMOD_LGUI', 'KMOD_RGUI', 'KMOD_GUI')

    def test_kmod__existence(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures KMOD constants exist.'
        for name in self.KMOD_CONSTANTS:
            self.assertTrue(hasattr(pygame.constants, name), f'missing constant {name}')

    def test_kmod__type(self):
        if False:
            print('Hello World!')
        'Ensures KMOD constants are the correct type.'
        for name in self.KMOD_CONSTANTS:
            value = getattr(pygame.constants, name)
            self.assertIs(type(value), int)

    def test_kmod__value_overlap(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures no unexpected KMOD constant values overlap.'
        EXPECTED_OVERLAPS = {frozenset(['KMOD_LGUI', 'KMOD_LMETA']), frozenset(['KMOD_RGUI', 'KMOD_RMETA']), frozenset(['KMOD_GUI', 'KMOD_META'])}
        overlaps = create_overlap_set(self.KMOD_CONSTANTS)
        self.assertSetEqual(overlaps, EXPECTED_OVERLAPS)

    def test_kmod__no_bitwise_overlap(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures certain KMOD constants have no overlapping bits.'
        NO_BITWISE_OVERLAP = ('KMOD_NONE', 'KMOD_LSHIFT', 'KMOD_RSHIFT', 'KMOD_LCTRL', 'KMOD_RCTRL', 'KMOD_LALT', 'KMOD_RALT', 'KMOD_LMETA', 'KMOD_RMETA', 'KMOD_NUM', 'KMOD_CAPS', 'KMOD_MODE')
        kmods = 0
        for name in NO_BITWISE_OVERLAP:
            value = getattr(pygame.constants, name)
            self.assertFalse(kmods & value)
            kmods |= value

    def test_kmod__bitwise_overlap(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures certain KMOD constants have overlapping bits.'
        KMOD_COMPRISED_DICT = {'KMOD_SHIFT': ('KMOD_LSHIFT', 'KMOD_RSHIFT'), 'KMOD_CTRL': ('KMOD_LCTRL', 'KMOD_RCTRL'), 'KMOD_ALT': ('KMOD_LALT', 'KMOD_RALT'), 'KMOD_META': ('KMOD_LMETA', 'KMOD_RMETA'), 'KMOD_GUI': ('KMOD_LGUI', 'KMOD_RGUI')}
        for (base_name, seq_names) in KMOD_COMPRISED_DICT.items():
            expected_value = 0
            for name in seq_names:
                expected_value |= getattr(pygame.constants, name)
            value = getattr(pygame.constants, base_name)
            self.assertEqual(value, expected_value)
if __name__ == '__main__':
    unittest.main()