"""
This is the Windows backend for keyboard events, and is implemented by
invoking the Win32 API through the ctypes module. This is error prone
and can introduce very unpythonic failure modes, such as segfaults and
low level memory leaks. But it is also dependency-free, very performant
well documented on Microsoft's website and scattered examples.

# TODO:
- Keypad numbers still print as numbers even when numlock is off.
- No way to specify if user wants a keypad key or not in `map_char`.
"""
from __future__ import unicode_literals
import re
import atexit
import traceback
from threading import Lock
from collections import defaultdict
from ._keyboard_event import KeyboardEvent, KEY_DOWN, KEY_UP
from ._canonical_names import normalize_name
try:
    chr = unichr
except NameError:
    pass
import ctypes
from ctypes import c_short, c_char, c_uint8, c_int32, c_int, c_uint, c_uint32, c_long, Structure, WINFUNCTYPE, POINTER
from ctypes.wintypes import WORD, DWORD, BOOL, HHOOK, MSG, LPWSTR, WCHAR, WPARAM, LPARAM, LONG, HMODULE, LPCWSTR, HINSTANCE, HWND
LPMSG = POINTER(MSG)
ULONG_PTR = POINTER(DWORD)
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
GetModuleHandleW = kernel32.GetModuleHandleW
GetModuleHandleW.restype = HMODULE
GetModuleHandleW.argtypes = [LPCWSTR]
user32 = ctypes.WinDLL('user32', use_last_error=True)
VK_PACKET = 231
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2
KEYEVENTF_KEYUP = 2
KEYEVENTF_UNICODE = 4

class KBDLLHOOKSTRUCT(Structure):
    _fields_ = [('vk_code', DWORD), ('scan_code', DWORD), ('flags', DWORD), ('time', c_int), ('dwExtraInfo', ULONG_PTR)]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (('dx', LONG), ('dy', LONG), ('mouseData', DWORD), ('dwFlags', DWORD), ('time', DWORD), ('dwExtraInfo', ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (('wVk', WORD), ('wScan', WORD), ('dwFlags', DWORD), ('time', DWORD), ('dwExtraInfo', ULONG_PTR))

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (('uMsg', DWORD), ('wParamL', WORD), ('wParamH', WORD))

class _INPUTunion(ctypes.Union):
    _fields_ = (('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT))

class INPUT(ctypes.Structure):
    _fields_ = (('type', DWORD), ('union', _INPUTunion))
LowLevelKeyboardProc = WINFUNCTYPE(c_int, WPARAM, LPARAM, POINTER(KBDLLHOOKSTRUCT))
SetWindowsHookEx = user32.SetWindowsHookExW
SetWindowsHookEx.argtypes = [c_int, LowLevelKeyboardProc, HINSTANCE, DWORD]
SetWindowsHookEx.restype = HHOOK
CallNextHookEx = user32.CallNextHookEx
CallNextHookEx.restype = c_int
UnhookWindowsHookEx = user32.UnhookWindowsHookEx
UnhookWindowsHookEx.argtypes = [HHOOK]
UnhookWindowsHookEx.restype = BOOL
GetMessage = user32.GetMessageW
GetMessage.argtypes = [LPMSG, HWND, c_uint, c_uint]
GetMessage.restype = BOOL
TranslateMessage = user32.TranslateMessage
TranslateMessage.argtypes = [LPMSG]
TranslateMessage.restype = BOOL
DispatchMessage = user32.DispatchMessageA
DispatchMessage.argtypes = [LPMSG]
keyboard_state_type = c_uint8 * 256
GetKeyboardState = user32.GetKeyboardState
GetKeyboardState.argtypes = [keyboard_state_type]
GetKeyboardState.restype = BOOL
GetKeyNameText = user32.GetKeyNameTextW
GetKeyNameText.argtypes = [c_long, LPWSTR, c_int]
GetKeyNameText.restype = c_int
MapVirtualKey = user32.MapVirtualKeyW
MapVirtualKey.argtypes = [c_uint, c_uint]
MapVirtualKey.restype = c_uint
ToUnicode = user32.ToUnicode
ToUnicode.argtypes = [c_uint, c_uint, keyboard_state_type, LPWSTR, c_int, c_uint]
ToUnicode.restype = c_int
SendInput = user32.SendInput
SendInput.argtypes = [c_uint, POINTER(INPUT), c_int]
SendInput.restype = c_uint
MAPVK_VK_TO_CHAR = 2
MAPVK_VK_TO_VSC = 0
MAPVK_VSC_TO_VK = 1
MAPVK_VK_TO_VSC_EX = 4
MAPVK_VSC_TO_VK_EX = 3
VkKeyScan = user32.VkKeyScanW
VkKeyScan.argtypes = [WCHAR]
VkKeyScan.restype = c_short
LLKHF_INJECTED = 16
WM_KEYDOWN = 256
WM_KEYUP = 257
WM_SYSKEYDOWN = 260
WM_SYSKEYUP = 261
keyboard_event_types = {WM_KEYDOWN: KEY_DOWN, WM_KEYUP: KEY_UP, WM_SYSKEYDOWN: KEY_DOWN, WM_SYSKEYUP: KEY_UP}
official_virtual_keys = {3: ('control-break processing', False), 8: ('backspace', False), 9: ('tab', False), 12: ('clear', False), 13: ('enter', False), 16: ('shift', False), 17: ('ctrl', False), 18: ('alt', False), 19: ('pause', False), 20: ('caps lock', False), 21: ('ime kana mode', False), 21: ('ime hanguel mode', False), 21: ('ime hangul mode', False), 23: ('ime junja mode', False), 24: ('ime final mode', False), 25: ('ime hanja mode', False), 25: ('ime kanji mode', False), 27: ('esc', False), 28: ('ime convert', False), 29: ('ime nonconvert', False), 30: ('ime accept', False), 31: ('ime mode change request', False), 32: ('spacebar', False), 33: ('page up', False), 34: ('page down', False), 35: ('end', False), 36: ('home', False), 37: ('left', False), 38: ('up', False), 39: ('right', False), 40: ('down', False), 41: ('select', False), 42: ('print', False), 43: ('execute', False), 44: ('print screen', False), 45: ('insert', False), 46: ('delete', False), 47: ('help', False), 48: ('0', False), 49: ('1', False), 50: ('2', False), 51: ('3', False), 52: ('4', False), 53: ('5', False), 54: ('6', False), 55: ('7', False), 56: ('8', False), 57: ('9', False), 65: ('a', False), 66: ('b', False), 67: ('c', False), 68: ('d', False), 69: ('e', False), 70: ('f', False), 71: ('g', False), 72: ('h', False), 73: ('i', False), 74: ('j', False), 75: ('k', False), 76: ('l', False), 77: ('m', False), 78: ('n', False), 79: ('o', False), 80: ('p', False), 81: ('q', False), 82: ('r', False), 83: ('s', False), 84: ('t', False), 85: ('u', False), 86: ('v', False), 87: ('w', False), 88: ('x', False), 89: ('y', False), 90: ('z', False), 91: ('left windows', False), 92: ('right windows', False), 93: ('applications', False), 95: ('sleep', False), 96: ('0', True), 97: ('1', True), 98: ('2', True), 99: ('3', True), 100: ('4', True), 101: ('5', True), 102: ('6', True), 103: ('7', True), 104: ('8', True), 105: ('9', True), 106: ('*', True), 107: ('+', True), 108: ('separator', True), 109: ('-', True), 110: ('decimal', True), 111: ('/', True), 112: ('f1', False), 113: ('f2', False), 114: ('f3', False), 115: ('f4', False), 116: ('f5', False), 117: ('f6', False), 118: ('f7', False), 119: ('f8', False), 120: ('f9', False), 121: ('f10', False), 122: ('f11', False), 123: ('f12', False), 124: ('f13', False), 125: ('f14', False), 126: ('f15', False), 127: ('f16', False), 128: ('f17', False), 129: ('f18', False), 130: ('f19', False), 131: ('f20', False), 132: ('f21', False), 133: ('f22', False), 134: ('f23', False), 135: ('f24', False), 144: ('num lock', False), 145: ('scroll lock', False), 160: ('left shift', False), 161: ('right shift', False), 162: ('left ctrl', False), 163: ('right ctrl', False), 164: ('left menu', False), 165: ('right menu', False), 166: ('browser back', False), 167: ('browser forward', False), 168: ('browser refresh', False), 169: ('browser stop', False), 170: ('browser search key', False), 171: ('browser favorites', False), 172: ('browser start and home', False), 173: ('volume mute', False), 174: ('volume down', False), 175: ('volume up', False), 176: ('next track', False), 177: ('previous track', False), 178: ('stop media', False), 179: ('play/pause media', False), 180: ('start mail', False), 181: ('select media', False), 182: ('start application 1', False), 183: ('start application 2', False), 187: ('+', False), 188: (',', False), 189: ('-', False), 190: ('.', False), 229: ('ime process', False), 246: ('attn', False), 247: ('crsel', False), 248: ('exsel', False), 249: ('erase eof', False), 250: ('play', False), 251: ('zoom', False), 252: ('reserved ', False), 253: ('pa1', False), 254: ('clear', False)}
tables_lock = Lock()
to_name = defaultdict(list)
from_name = defaultdict(list)
scan_code_to_vk = {}
distinct_modifiers = [(), ('shift',), ('alt gr',), ('num lock',), ('shift', 'num lock'), ('caps lock',), ('shift', 'caps lock'), ('alt gr', 'num lock')]
name_buffer = ctypes.create_unicode_buffer(32)
unicode_buffer = ctypes.create_unicode_buffer(32)
keyboard_state = keyboard_state_type()

def get_event_names(scan_code, vk, is_extended, modifiers):
    if False:
        i = 10
        return i + 15
    is_keypad = (scan_code, vk, is_extended) in keypad_keys
    is_official = vk in official_virtual_keys
    if is_keypad and is_official:
        yield official_virtual_keys[vk][0]
    keyboard_state[16] = 128 * ('shift' in modifiers)
    keyboard_state[17] = 128 * ('alt gr' in modifiers)
    keyboard_state[18] = 128 * ('alt gr' in modifiers)
    keyboard_state[20] = 1 * ('caps lock' in modifiers)
    keyboard_state[144] = 1 * ('num lock' in modifiers)
    keyboard_state[145] = 1 * ('scroll lock' in modifiers)
    unicode_ret = ToUnicode(vk, scan_code, keyboard_state, unicode_buffer, len(unicode_buffer), 0)
    if unicode_ret and unicode_buffer.value:
        yield unicode_buffer.value
        ToUnicode(vk, scan_code, keyboard_state, unicode_buffer, len(unicode_buffer), 0)
    name_ret = GetKeyNameText(scan_code << 16 | is_extended << 24, name_buffer, 1024)
    if name_ret and name_buffer.value:
        yield name_buffer.value
    char = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_CHAR) & 255
    if char != 0:
        yield chr(char)
    if not is_keypad and is_official:
        yield official_virtual_keys[vk][0]

def _setup_name_tables():
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensures the scan code/virtual key code/name translation tables are\n    filled.\n    '
    with tables_lock:
        if to_name:
            return
        all_scan_codes = [(sc, user32.MapVirtualKeyExW(sc, MAPVK_VSC_TO_VK_EX, 0)) for sc in range(256)]
        all_vks = [(user32.MapVirtualKeyExW(vk, MAPVK_VK_TO_VSC_EX, 0), vk) for vk in range(256)]
        for (scan_code, vk) in all_scan_codes + all_vks:
            if (scan_code, vk, 0, 0, 0) in to_name:
                continue
            if scan_code not in scan_code_to_vk:
                scan_code_to_vk[scan_code] = vk
            for extended in [0, 1]:
                for modifiers in distinct_modifiers:
                    entry = (scan_code, vk, extended, modifiers)
                    names = list(get_event_names(*entry))
                    if names:
                        lowercase_names = [name.lower() for name in names]
                        to_name[entry] = names + lowercase_names
                        for (i, name) in enumerate(map(normalize_name, names + lowercase_names)):
                            from_name[name].append((i, entry))
        for extended in [0, 1]:
            for modifiers in distinct_modifiers:
                to_name[541, 162, extended, modifiers] = ['alt gr']
                from_name['alt gr'].append((1, (541, 162, extended, modifiers)))
    modifiers_preference = defaultdict(lambda : 10)
    modifiers_preference.update({(): 0, ('shift',): 1, ('alt gr',): 2, ('ctrl',): 3, ('alt',): 4})

    def order_key(line):
        if False:
            print('Hello World!')
        (i, entry) = line
        (scan_code, vk, extended, modifiers) = entry
        return (modifiers_preference[modifiers], i, extended, vk, scan_code)
    for (name, entries) in list(from_name.items()):
        from_name[name] = sorted(set(entries), key=order_key)
init = _setup_name_tables
keypad_keys = [(126, 194, 0), (126, 194, 0), (28, 13, 1), (28, 13, 1), (53, 111, 1), (53, 111, 1), (55, 106, 0), (55, 106, 0), (69, 144, 1), (69, 144, 1), (71, 103, 0), (71, 36, 0), (72, 104, 0), (72, 38, 0), (73, 105, 0), (73, 33, 0), (74, 109, 0), (74, 109, 0), (75, 100, 0), (75, 37, 0), (76, 101, 0), (76, 12, 0), (77, 102, 0), (77, 39, 0), (78, 107, 0), (78, 107, 0), (79, 35, 0), (79, 97, 0), (80, 40, 0), (80, 98, 0), (81, 34, 0), (81, 99, 0), (82, 45, 0), (82, 96, 0), (83, 110, 0), (83, 46, 0)]
shift_is_pressed = False
altgr_is_pressed = False
ignore_next_right_alt = False
shift_vks = set([16, 160, 161])

def prepare_intercept(callback):
    if False:
        return 10
    '\n    Registers a Windows low level keyboard hook. The provided callback will\n    be invoked for each high-level keyboard event, and is expected to return\n    True if the key event should be passed to the next program, or False if\n    the event is to be blocked.\n\n    No event is processed until the Windows messages are pumped (see\n    start_intercept).\n    '
    _setup_name_tables()

    def process_key(event_type, vk, scan_code, is_extended):
        if False:
            while True:
                i = 10
        global shift_is_pressed, altgr_is_pressed, ignore_next_right_alt
        if vk == 165 and ignore_next_right_alt:
            ignore_next_right_alt = False
            return True
        modifiers = ('shift',) * shift_is_pressed + ('alt gr',) * altgr_is_pressed + ('num lock',) * (user32.GetKeyState(144) & 1) + ('caps lock',) * (user32.GetKeyState(20) & 1) + ('scroll lock',) * (user32.GetKeyState(145) & 1)
        entry = (scan_code, vk, is_extended, modifiers)
        if entry not in to_name:
            to_name[entry] = list(get_event_names(*entry))
        names = to_name[entry]
        name = names[0] if names else None
        if vk in shift_vks:
            shift_is_pressed = event_type == KEY_DOWN
        if scan_code == 541 and vk == 162:
            ignore_next_right_alt = True
            altgr_is_pressed = event_type == KEY_DOWN
        is_keypad = (scan_code, vk, is_extended) in keypad_keys
        return callback(KeyboardEvent(event_type=event_type, scan_code=scan_code or -vk, name=name, is_keypad=is_keypad))

    def low_level_keyboard_handler(nCode, wParam, lParam):
        if False:
            i = 10
            return i + 15
        try:
            vk = lParam.contents.vk_code
            fake_alt = LLKHF_INJECTED | 32
            if vk != VK_PACKET and lParam.contents.flags & fake_alt != fake_alt:
                event_type = KEY_UP if wParam & 1 else KEY_DOWN
                is_extended = lParam.contents.flags & 1
                scan_code = lParam.contents.scan_code
                should_continue = process_key(event_type, vk, scan_code, is_extended)
                if not should_continue:
                    return -1
        except Exception as e:
            print('Error in keyboard hook:')
            traceback.print_exc()
        return CallNextHookEx(None, nCode, wParam, lParam)
    WH_KEYBOARD_LL = c_int(13)
    keyboard_callback = LowLevelKeyboardProc(low_level_keyboard_handler)
    handle = GetModuleHandleW(None)
    thread_id = DWORD(0)
    keyboard_hook = SetWindowsHookEx(WH_KEYBOARD_LL, keyboard_callback, handle, thread_id)
    atexit.register(UnhookWindowsHookEx, keyboard_callback)

def listen(callback):
    if False:
        return 10
    prepare_intercept(callback)
    msg = LPMSG()
    while not GetMessage(msg, 0, 0, 0):
        TranslateMessage(msg)
        DispatchMessage(msg)

def map_name(name):
    if False:
        print('Hello World!')
    _setup_name_tables()
    entries = from_name.get(name)
    if not entries:
        raise ValueError('Key name {} is not mapped to any known key.'.format(repr(name)))
    for (i, entry) in entries:
        (scan_code, vk, is_extended, modifiers) = entry
        yield (scan_code or -vk, modifiers)

def _send_event(code, event_type):
    if False:
        return 10
    if code == 541:
        user32.keybd_event(17, code, event_type, 0)
        user32.keybd_event(18, code, event_type, 0)
    elif code > 0:
        vk = scan_code_to_vk.get(code, 0)
        user32.keybd_event(vk, code, event_type, 0)
    else:
        user32.keybd_event(-code, 0, event_type, 0)

def press(code):
    if False:
        print('Hello World!')
    _send_event(code, 0)

def release(code):
    if False:
        i = 10
        return i + 15
    _send_event(code, 2)

def type_unicode(character):
    if False:
        return 10
    surrogates = bytearray(character.encode('utf-16le'))
    presses = []
    releases = []
    for i in range(0, len(surrogates), 2):
        (higher, lower) = surrogates[i:i + 2]
        structure = KEYBDINPUT(0, (lower << 8) + higher, KEYEVENTF_UNICODE, 0, None)
        presses.append(INPUT(INPUT_KEYBOARD, _INPUTunion(ki=structure)))
        structure = KEYBDINPUT(0, (lower << 8) + higher, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, None)
        releases.append(INPUT(INPUT_KEYBOARD, _INPUTunion(ki=structure)))
    inputs = presses + releases
    nInputs = len(inputs)
    LPINPUT = INPUT * nInputs
    pInputs = LPINPUT(*inputs)
    cbSize = c_int(ctypes.sizeof(INPUT))
    SendInput(nInputs, pInputs, cbSize)
if __name__ == '__main__':
    _setup_name_tables()
    import pprint
    pprint.pprint(to_name)
    pprint.pprint(from_name)