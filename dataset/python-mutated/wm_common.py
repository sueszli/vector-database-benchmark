"""
Common definitions for a Windows provider
=========================================

This file provides common definitions for constants used by WM_Touch / WM_Pen.
"""
import os
WM_MOUSEFIRST = 512
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
WM_MOUSELAST = 522
WM_DPICHANGED = 736
WM_GETDPISCALEDSIZE = 740
WM_NCCALCSIZE = 131
WM_TOUCH = 576
TOUCHEVENTF_MOVE = 1
TOUCHEVENTF_DOWN = 2
TOUCHEVENTF_UP = 4
PEN_OR_TOUCH_SIGNATURE = 4283520768
PEN_OR_TOUCH_MASK = 4294967040
PEN_EVENT_TOUCH_MASK = 128
SM_CYCAPTION = 4
WM_TABLET_QUERYSYSTEMGESTURE = 716
TABLET_DISABLE_PRESSANDHOLD = 1
TABLET_DISABLE_PENTAPFEEDBACK = 8
TABLET_DISABLE_PENBARRELFEEDBACK = 16
TABLET_DISABLE_TOUCHUIFORCEON = 256
TABLET_DISABLE_TOUCHUIFORCEOFF = 512
TABLET_DISABLE_TOUCHSWITCH = 32768
TABLET_DISABLE_FLICKS = 65536
TABLET_ENABLE_FLICKSONCONTEXT = 131072
TABLET_ENABLE_FLICKLEARNINGMODE = 262144
TABLET_DISABLE_SMOOTHSCROLLING = 524288
TABLET_DISABLE_FLICKFALLBACKKEYS = 1048576
GWL_WNDPROC = -4
QUERYSYSTEMGESTURE_WNDPROC = TABLET_DISABLE_PRESSANDHOLD | TABLET_DISABLE_PENTAPFEEDBACK | TABLET_DISABLE_PENBARRELFEEDBACK | TABLET_DISABLE_SMOOTHSCROLLING | TABLET_DISABLE_FLICKFALLBACKKEYS | TABLET_DISABLE_TOUCHSWITCH | TABLET_DISABLE_FLICKS
if 'KIVY_DOC' not in os.environ:
    from ctypes.wintypes import ULONG, HANDLE, DWORD, LONG, UINT, WPARAM, LPARAM, BOOL, HWND, POINT, RECT as RECT_BASE
    from ctypes import windll, WINFUNCTYPE, POINTER, c_int, c_longlong, c_void_p, Structure, sizeof, byref, cast

    class RECT(RECT_BASE):
        x = property(lambda self: self.left)
        y = property(lambda self: self.top)
        w = property(lambda self: self.right - self.left)
        h = property(lambda self: self.bottom - self.top)
    if not hasattr(windll.user32, 'RegisterTouchWindow'):
        raise Exception('Unsupported Window version')
    LRESULT = LPARAM
    WNDPROC = WINFUNCTYPE(LRESULT, HWND, UINT, WPARAM, LPARAM)

    class TOUCHINPUT(Structure):
        _fields_ = [('x', LONG), ('y', LONG), ('pSource', HANDLE), ('id', DWORD), ('flags', DWORD), ('mask', DWORD), ('time', DWORD), ('extraInfo', POINTER(ULONG)), ('size_x', DWORD), ('size_y', DWORD)]

        def size(self):
            if False:
                return 10
            return (self.size_x, self.size_y)

        def screen_x(self):
            if False:
                i = 10
                return i + 15
            return self.x / 100.0

        def screen_y(self):
            if False:
                return 10
            return self.y / 100.0

        def _event_type(self):
            if False:
                while True:
                    i = 10
            if self.flags & TOUCHEVENTF_MOVE:
                return 'update'
            if self.flags & TOUCHEVENTF_DOWN:
                return 'begin'
            if self.flags & TOUCHEVENTF_UP:
                return 'end'
        event_type = property(_event_type)

    def SetWindowLong_WndProc_wrapper_generator(func):
        if False:
            print('Hello World!')

        def _closure(hWnd, wndProc):
            if False:
                while True:
                    i = 10
            oldAddr = func(hWnd, GWL_WNDPROC, cast(wndProc, c_void_p).value)
            return cast(c_void_p(oldAddr), WNDPROC)
        return _closure
    try:
        LONG_PTR = c_longlong
        windll.user32.SetWindowLongPtrW.restype = LONG_PTR
        windll.user32.SetWindowLongPtrW.argtypes = [HWND, c_int, LONG_PTR]
        SetWindowLong_WndProc_wrapper = SetWindowLong_WndProc_wrapper_generator(windll.user32.SetWindowLongPtrW)
    except AttributeError:
        windll.user32.SetWindowLongW.restype = LONG
        windll.user32.SetWindowLongW.argtypes = [HWND, c_int, LONG]
        SetWindowLong_WndProc_wrapper = SetWindowLong_WndProc_wrapper_generator(windll.user32.SetWindowLongW)
    windll.user32.GetMessageExtraInfo.restype = LPARAM
    windll.user32.GetMessageExtraInfo.argtypes = []
    windll.user32.GetClientRect.restype = BOOL
    windll.user32.GetClientRect.argtypes = [HANDLE, POINTER(RECT_BASE)]
    windll.user32.GetWindowRect.restype = BOOL
    windll.user32.GetWindowRect.argtypes = [HANDLE, POINTER(RECT_BASE)]
    windll.user32.CallWindowProcW.restype = LRESULT
    windll.user32.CallWindowProcW.argtypes = [WNDPROC, HWND, UINT, WPARAM, LPARAM]
    windll.user32.GetActiveWindow.restype = HWND
    windll.user32.GetActiveWindow.argtypes = []
    windll.user32.RegisterTouchWindow.restype = BOOL
    windll.user32.RegisterTouchWindow.argtypes = [HWND, ULONG]
    windll.user32.UnregisterTouchWindow.restype = BOOL
    windll.user32.UnregisterTouchWindow.argtypes = [HWND]
    windll.user32.GetTouchInputInfo.restype = BOOL
    windll.user32.GetTouchInputInfo.argtypes = [HANDLE, UINT, POINTER(TOUCHINPUT), c_int]
    windll.user32.GetSystemMetrics.restype = c_int
    windll.user32.GetSystemMetrics.argtypes = [c_int]
    windll.user32.ClientToScreen.restype = BOOL
    windll.user32.ClientToScreen.argtypes = [HWND, POINTER(POINT)]
    try:
        windll.user32.GetDpiForWindow.restype = UINT
        windll.user32.GetDpiForWindow.argtypes = [HWND]
    except AttributeError:
        pass