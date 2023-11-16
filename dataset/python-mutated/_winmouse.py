import ctypes
import time
from ctypes import c_short, c_char, c_uint8, c_int32, c_int, c_uint, c_uint32, c_long, byref, Structure, CFUNCTYPE, POINTER
from ctypes.wintypes import DWORD, BOOL, HHOOK, MSG, LPWSTR, WCHAR, WPARAM, LPARAM
LPMSG = POINTER(MSG)
import atexit
from ._mouse_event import ButtonEvent, WheelEvent, MoveEvent, LEFT, RIGHT, MIDDLE, X, X2, UP, DOWN, DOUBLE, WHEEL, HORIZONTAL, VERTICAL
user32 = ctypes.WinDLL('user32', use_last_error=True)

class MSLLHOOKSTRUCT(Structure):
    _fields_ = [('x', c_long), ('y', c_long), ('data', c_int32), ('reserved', c_int32), ('flags', DWORD), ('time', c_int)]
LowLevelMouseProc = CFUNCTYPE(c_int, WPARAM, LPARAM, POINTER(MSLLHOOKSTRUCT))
SetWindowsHookEx = user32.SetWindowsHookExA
SetWindowsHookEx.restype = HHOOK
CallNextHookEx = user32.CallNextHookEx
CallNextHookEx.restype = c_int
UnhookWindowsHookEx = user32.UnhookWindowsHookEx
UnhookWindowsHookEx.argtypes = [HHOOK]
UnhookWindowsHookEx.restype = BOOL
GetMessage = user32.GetMessageW
GetMessage.argtypes = [LPMSG, c_int, c_int, c_int]
GetMessage.restype = BOOL
TranslateMessage = user32.TranslateMessage
TranslateMessage.argtypes = [LPMSG]
TranslateMessage.restype = BOOL
DispatchMessage = user32.DispatchMessageA
DispatchMessage.argtypes = [LPMSG]
WM_MOUSEMOVE = 512
WM_LBUTTONDOWN = 513
WM_LBUTTONUP = 514
WM_LBUTTONDBLCLK = 515
WM_RBUTTONDOWN = 516
WM_RBUTTONUP = 517
WM_RBUTTONDBLCLK = 518
WM_MBUTTONDOWN = 519
WM_MBUTTONUP = 520
WM_MBUTTONDBLCLK = 521
WM_MOUSEWHEEL = 522
WM_XBUTTONDOWN = 523
WM_XBUTTONUP = 524
WM_XBUTTONDBLCLK = 525
WM_NCXBUTTONDOWN = 171
WM_NCXBUTTONUP = 172
WM_NCXBUTTONDBLCLK = 173
WM_MOUSEHWHEEL = 526
WM_LBUTTONDOWN = 513
WM_LBUTTONUP = 514
WM_MOUSEMOVE = 512
WM_MOUSEWHEEL = 522
WM_MOUSEHWHEEL = 526
WM_RBUTTONDOWN = 516
WM_RBUTTONUP = 517
buttons_by_wm_code = {WM_LBUTTONDOWN: (DOWN, LEFT), WM_LBUTTONUP: (UP, LEFT), WM_LBUTTONDBLCLK: (DOUBLE, LEFT), WM_RBUTTONDOWN: (DOWN, RIGHT), WM_RBUTTONUP: (UP, RIGHT), WM_RBUTTONDBLCLK: (DOUBLE, RIGHT), WM_MBUTTONDOWN: (DOWN, MIDDLE), WM_MBUTTONUP: (UP, MIDDLE), WM_MBUTTONDBLCLK: (DOUBLE, MIDDLE), WM_XBUTTONDOWN: (DOWN, X), WM_XBUTTONUP: (UP, X), WM_XBUTTONDBLCLK: (DOUBLE, X)}
MOUSEEVENTF_ABSOLUTE = 32768
MOUSEEVENTF_MOVE = 1
MOUSEEVENTF_WHEEL = 2048
MOUSEEVENTF_HWHEEL = 4096
MOUSEEVENTF_LEFTDOWN = 2
MOUSEEVENTF_LEFTUP = 4
MOUSEEVENTF_RIGHTDOWN = 8
MOUSEEVENTF_RIGHTUP = 16
MOUSEEVENTF_MIDDLEDOWN = 32
MOUSEEVENTF_MIDDLEUP = 64
MOUSEEVENTF_XDOWN = 128
MOUSEEVENTF_XUP = 256
simulated_mouse_codes = {(WHEEL, HORIZONTAL): MOUSEEVENTF_HWHEEL, (WHEEL, VERTICAL): MOUSEEVENTF_WHEEL, (DOWN, LEFT): MOUSEEVENTF_LEFTDOWN, (UP, LEFT): MOUSEEVENTF_LEFTUP, (DOWN, RIGHT): MOUSEEVENTF_RIGHTDOWN, (UP, RIGHT): MOUSEEVENTF_RIGHTUP, (DOWN, MIDDLE): MOUSEEVENTF_MIDDLEDOWN, (UP, MIDDLE): MOUSEEVENTF_MIDDLEUP, (DOWN, X): MOUSEEVENTF_XDOWN, (UP, X): MOUSEEVENTF_XUP}
NULL = c_int(0)
WHEEL_DELTA = 120
init = lambda : None

def listen(queue):
    if False:
        i = 10
        return i + 15

    def low_level_mouse_handler(nCode, wParam, lParam):
        if False:
            for i in range(10):
                print('nop')
        struct = lParam.contents
        t = time.time()
        if wParam == WM_MOUSEMOVE:
            event = MoveEvent(struct.x, struct.y, t)
        elif wParam == WM_MOUSEWHEEL:
            event = WheelEvent(struct.data / (WHEEL_DELTA * (2 << 15)), t)
        elif wParam in buttons_by_wm_code:
            (type, button) = buttons_by_wm_code.get(wParam, ('?', '?'))
            if wParam >= WM_XBUTTONDOWN:
                button = {65536: X, 131072: X2}[struct.data]
            event = ButtonEvent(type, button, t)
        queue.put(event)
        return CallNextHookEx(NULL, nCode, wParam, lParam)
    WH_MOUSE_LL = c_int(14)
    mouse_callback = LowLevelMouseProc(low_level_mouse_handler)
    mouse_hook = SetWindowsHookEx(WH_MOUSE_LL, mouse_callback, NULL, NULL)
    atexit.register(UnhookWindowsHookEx, mouse_hook)
    msg = LPMSG()
    while not GetMessage(msg, NULL, NULL, NULL):
        TranslateMessage(msg)
        DispatchMessage(msg)

def _translate_button(button):
    if False:
        return 10
    if button == X or button == X2:
        return (X, {X: 65536, X2: 131072}[button])
    else:
        return (button, 0)

def press(button=LEFT):
    if False:
        print('Hello World!')
    (button, data) = _translate_button(button)
    code = simulated_mouse_codes[DOWN, button]
    user32.mouse_event(code, 0, 0, data, 0)

def release(button=LEFT):
    if False:
        print('Hello World!')
    (button, data) = _translate_button(button)
    code = simulated_mouse_codes[UP, button]
    user32.mouse_event(code, 0, 0, data, 0)

def wheel(delta=1):
    if False:
        for i in range(10):
            print('nop')
    code = simulated_mouse_codes[WHEEL, VERTICAL]
    user32.mouse_event(code, 0, 0, int(delta * WHEEL_DELTA), 0)

def move_to(x, y):
    if False:
        i = 10
        return i + 15
    user32.SetCursorPos(int(x), int(y))

def move_relative(x, y):
    if False:
        i = 10
        return i + 15
    user32.mouse_event(MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)

class POINT(Structure):
    _fields_ = [('x', c_long), ('y', c_long)]

def get_position():
    if False:
        for i in range(10):
            print('nop')
    point = POINT()
    user32.GetCursorPos(byref(point))
    return (point.x, point.y)
if __name__ == '__main__':

    def p(e):
        if False:
            while True:
                i = 10
        print(e)
    listen(p)