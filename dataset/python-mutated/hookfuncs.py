from ctypes import byref, WinDLL, WinError, sizeof, pointer, c_int, c_ulong, c_void_p, c_wchar_p, c_uint, CFUNCTYPE, cast, c_void_p as LPRECT, c_void_p as PSID, create_string_buffer, create_unicode_buffer, Structure, POINTER, WINFUNCTYPE
from ctypes.wintypes import BOOL, DOUBLE, DWORD, HBITMAP, HINSTANCE, HDC, HGDIOBJ, HANDLE, HWND, INT, LPARAM, LONG, RECT, UINT, WORD, HMODULE, HHOOK, POINT, LPCWSTR, WPARAM, LPVOID, LPSTR, LPWSTR, BYTE, WCHAR, SHORT, WPARAM as ULONG_PTR, LPARAM as LRESULT

class MSG(Structure):
    _fields_ = [('hWnd', HWND), ('message', c_uint), ('wParam', WPARAM), ('lParam', LPARAM), ('time', DWORD), ('pt', POINT), ('private', DWORD)]
WH_MOUSE_LL = 14
WM_MOUSEFIRST = 512
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
SRCCOPY = 13369376
DIB_RGB_COLORS = 0
WM_KEYDOWN = 256
WM_KEYUP = 257
WM_SYSKEYDOWN = 260
WM_SYSKEYUP = 261
WH_KEYBOARD_LL = 13
VK_CAPITAL = 20
VK_SHIFT = 16
VK_LSHIFT = 160
VK_RSHIFT = 161
VK_CONTROL = 17
VK_LCONTROL = 162
VK_RCONTROL = 163
VK_MENU = 18
VK_LMENU = 164
VK_RMENU = 165
VK_RETURN = 13
VK_ESCAPE = 27
VK_LWIN = 91
VK_RWIN = 92
LPMSG = POINTER(MSG)

def LOWORD(x):
    if False:
        print('Hello World!')
    return x & 65535
HOOKPROC = WINFUNCTYPE(LRESULT, c_int, WPARAM, LPARAM)
LOWLEVELKEYBOARDPROC = CFUNCTYPE(LRESULT, c_int, WPARAM, LPARAM)
MONITORENUMPROC = WINFUNCTYPE(INT, DWORD, DWORD, POINTER(RECT), DOUBLE)

class KBDLLHOOKSTRUCT(Structure):
    _fields_ = [('vkCode', DWORD), ('scanCode', DWORD), ('flags', DWORD), ('time', DWORD), ('dwExtraInfo', ULONG_PTR)]

class BITMAPINFOHEADER(Structure):
    _fields_ = [('biSize', DWORD), ('biWidth', LONG), ('biHeight', LONG), ('biPlanes', WORD), ('biBitCount', WORD), ('biCompression', DWORD), ('biSizeImage', DWORD), ('biXPelsPerMeter', LONG), ('biYPelsPerMeter', LONG), ('biClrUsed', DWORD), ('biClrImportant', DWORD)]

class BITMAPINFO(Structure):
    _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', DWORD * 3)]

class POINT(Structure):
    _fields_ = [('x', c_ulong), ('y', c_ulong)]
user32 = WinDLL('user32')
kernel32 = WinDLL('kernel32')
gdi32 = WinDLL('gdi32')
psapi = WinDLL('psapi')
GetForegroundWindow = user32.GetForegroundWindow
GetForegroundWindow.restype = HWND
GetForegroundWindow.argtypes = []
GetCurrentProcess = kernel32.GetCurrentProcess
GetCurrentProcess.restype = HANDLE
GetCurrentProcess.argtypes = []
GetCurrentProcessId = kernel32.GetCurrentProcessId
GetCurrentProcessId.restype = DWORD
GetCurrentProcessId.argtypes = []
OpenProcess = kernel32.OpenProcess
OpenProcess.restype = HANDLE
OpenProcess.argtypes = [DWORD, BOOL, DWORD]
LocalAlloc = kernel32.LocalAlloc
LocalAlloc.restype = HANDLE
LocalAlloc.argtypes = [PSID, DWORD]
LocalFree = kernel32.LocalFree
LocalFree.restype = HANDLE
LocalFree.argtypes = [HANDLE]
GetModuleBaseNameW = psapi.GetModuleBaseNameW
GetModuleBaseNameW.restype = DWORD
GetModuleBaseNameW.argtypes = [HWND, HMODULE, c_void_p, DWORD]
GetModuleHandleW = kernel32.GetModuleHandleW
GetModuleHandleW.restype = HMODULE
GetModuleHandleW.argtypes = [LPCWSTR]
GetWindowThreadProcessId = user32.GetWindowThreadProcessId
GetWindowThreadProcessId.restype = DWORD
GetWindowThreadProcessId.argtypes = (HWND, POINTER(DWORD))
SetTimer = user32.SetTimer
SetTimer.restype = ULONG_PTR
SetTimer.argtypes = (HWND, ULONG_PTR, UINT, c_void_p)
KillTimer = user32.KillTimer
KillTimer.restype = BOOL
KillTimer.argtypes = (HWND, ULONG_PTR)
SetWindowsHookEx = user32.SetWindowsHookExW
SetWindowsHookEx.argtypes = (c_int, HOOKPROC, HINSTANCE, DWORD)
SetWindowsHookEx.restype = HHOOK
UnhookWindowsHookEx = user32.UnhookWindowsHookEx
UnhookWindowsHookEx.restype = BOOL
UnhookWindowsHookEx.argtypes = [HHOOK]
CallNextHookEx = user32.CallNextHookEx
CallNextHookEx.restype = LRESULT
CallNextHookEx.argtypes = (HHOOK, c_int, WPARAM, LPARAM)
GetMessageW = user32.GetMessageW
GetMessageW.argtypes = (LPMSG, HWND, UINT, UINT)
CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [HWND]
GetKeyboardState = user32.GetKeyboardState
GetKeyboardLayout = user32.GetKeyboardLayout
ToUnicodeEx = user32.ToUnicodeEx
ToAsciiEx = user32.ToAsciiEx
GetKeyState = user32.GetKeyState
GetKeyState.restype = SHORT
GetKeyState.argtypes = [INT]
OpenClipboard = user32.OpenClipboard
CloseClipboard = user32.CloseClipboard
GetClipboardData = user32.GetClipboardData
GetCursorPos = user32.GetCursorPos
GetSystemMetrics = user32.GetSystemMetrics
GetSystemMetrics.argtypes = [INT]
GetSystemMetrics.restypes = INT
EnumDisplayMonitors = user32.EnumDisplayMonitors
EnumDisplayMonitors.argtypes = [HDC, LPRECT, MONITORENUMPROC, LPARAM]
EnumDisplayMonitors.restypes = BOOL
GetWindowDC = user32.GetWindowDC
GetWindowDC.argtypes = [HWND]
GetWindowDC.restypes = HDC
GetWindowText = user32.GetWindowTextW
GetWindowText.argtypes = (HWND, LPWSTR, INT)
GetWindowText.restype = c_int
CreateCompatibleDC = gdi32.CreateCompatibleDC
CreateCompatibleDC.argtypes = [HDC]
CreateCompatibleDC.restypes = HDC
CreateCompatibleBitmap = gdi32.CreateCompatibleBitmap
CreateCompatibleBitmap.argtypes = [HDC, INT, INT]
CreateCompatibleBitmap.restypes = HBITMAP
SelectObject = gdi32.SelectObject
SelectObject.argtypes = [HDC, HGDIOBJ]
SelectObject.restypes = HGDIOBJ
BitBlt = gdi32.BitBlt
BitBlt.argtypes = [HDC, INT, INT, INT, INT, HDC, INT, INT, DWORD]
BitBlt.restypes = BOOL
GetDIBits = gdi32.GetDIBits
GetDIBits.restypes = INT
GetDIBits.argtypes = [HDC, HBITMAP, UINT, UINT, LPVOID, POINTER(BITMAPINFO), UINT]
DeleteObject = gdi32.DeleteObject
DeleteObject.argtypes = [HGDIOBJ]
DeleteObject.restypes = BOOL

def get_current_process():
    if False:
        print('Hello World!')
    hwnd = GetForegroundWindow()
    pid = c_ulong(0)
    GetWindowThreadProcessId(hwnd, byref(pid))
    executable = create_unicode_buffer('\x00', 512)
    h_process = OpenProcess(1024 | 16, False, pid)
    GetModuleBaseNameW(h_process, None, byref(executable), 512)
    window_title = create_unicode_buffer('\x00', 512)
    lpBuffer = cast(byref(window_title), LPWSTR)
    GetWindowText(hwnd, lpBuffer, 512)
    CloseHandle(hwnd)
    CloseHandle(h_process)
    return (executable.value, window_title.value)

def get_clipboard():
    if False:
        return 10
    OpenClipboard(0)
    pcontents = GetClipboardData(13)
    data = c_wchar_p(pcontents).value
    CloseClipboard()
    return data

def get_mouse_xy():
    if False:
        for i in range(10):
            print('nop')
    pt = POINT()
    GetCursorPos(byref(pt))
    return (pt.x, pt.y)