"""
This file is based on the code from https://github.com/JustAMan/pyWinClobber/blob/master/win32elevate.py

Copyright (c) 2013 by JustAMan at GitHub

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import ctypes
import os
import subprocess
from ctypes import POINTER, byref, c_char_p, c_int, c_ulong, c_void_p
from ctypes.wintypes import BOOL, DWORD, HANDLE, HINSTANCE, HKEY, HWND, LPCSTR, LPCWSTR, LPDWORD, SHORT, SMALL_RECT, WORD
from xonsh import lazyimps, platform
from xonsh.lazyasd import lazyobject
__all__ = ('sudo',)

@lazyobject
def CloseHandle():
    if False:
        print('Hello World!')
    ch = ctypes.windll.kernel32.CloseHandle
    ch.argtypes = (HANDLE,)
    ch.restype = BOOL
    return ch

@lazyobject
def GetActiveWindow():
    if False:
        for i in range(10):
            print('nop')
    gaw = ctypes.windll.user32.GetActiveWindow
    gaw.argtypes = ()
    gaw.restype = HANDLE
    return gaw
TOKEN_READ = 131080

class ShellExecuteInfo(ctypes.Structure):
    _fields_ = [('cbSize', DWORD), ('fMask', c_ulong), ('hwnd', HWND), ('lpVerb', c_char_p), ('lpFile', c_char_p), ('lpParameters', c_char_p), ('lpDirectory', c_char_p), ('nShow', c_int), ('hInstApp', HINSTANCE), ('lpIDList', c_void_p), ('lpClass', c_char_p), ('hKeyClass', HKEY), ('dwHotKey', DWORD), ('hIcon', HANDLE), ('hProcess', HANDLE)]

    def __init__(self, **kw):
        if False:
            for i in range(10):
                print('nop')
        ctypes.Structure.__init__(self)
        self.cbSize = ctypes.sizeof(self)
        for (field_name, field_value) in kw.items():
            setattr(self, field_name, field_value)

@lazyobject
def ShellExecuteEx():
    if False:
        for i in range(10):
            print('nop')
    see = ctypes.windll.Shell32.ShellExecuteExA
    PShellExecuteInfo = ctypes.POINTER(ShellExecuteInfo)
    see.argtypes = (PShellExecuteInfo,)
    see.restype = BOOL
    return see

@lazyobject
def WaitForSingleObject():
    if False:
        i = 10
        return i + 15
    wfso = ctypes.windll.kernel32.WaitForSingleObject
    wfso.argtypes = (HANDLE, DWORD)
    wfso.restype = DWORD
    return wfso
SW_SHOW = 5
SEE_MASK_NOCLOSEPROCESS = 64
SEE_MASK_NO_CONSOLE = 32768
INFINITE = -1

def wait_and_close_handle(process_handle):
    if False:
        return 10
    '\n    Waits till spawned process finishes and closes the handle for it\n\n    Parameters\n    ----------\n    process_handle : HANDLE\n        The Windows handle for the process\n    '
    WaitForSingleObject(process_handle, INFINITE)
    CloseHandle(process_handle)

def sudo(executable, args=None):
    if False:
        i = 10
        return i + 15
    '\n    This will re-run current Python script requesting to elevate administrative rights.\n\n    Parameters\n    ----------\n    executable : str\n        The path/name of the executable\n    args : list of str\n        The arguments to be passed to the executable\n    '
    if not args:
        args = []
    execute_info = ShellExecuteInfo(fMask=SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NO_CONSOLE, hwnd=GetActiveWindow(), lpVerb=b'runas', lpFile=executable.encode('utf-8'), lpParameters=subprocess.list2cmdline(args).encode('utf-8'), lpDirectory=None, nShow=SW_SHOW)
    if not ShellExecuteEx(byref(execute_info)):
        raise ctypes.WinError()
    wait_and_close_handle(execute_info.hProcess)
ENABLE_PROCESSED_INPUT = 1
ENABLE_LINE_INPUT = 2
ENABLE_ECHO_INPUT = 4
ENABLE_WINDOW_INPUT = 8
ENABLE_MOUSE_INPUT = 16
ENABLE_INSERT_MODE = 32
ENABLE_QUICK_EDIT_MODE = 64
ENABLE_PROCESSED_OUTPUT = 1
ENABLE_WRAP_AT_EOL_OUTPUT = 2
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 4

def check_zero(result, func, args):
    if False:
        i = 10
        return i + 15
    if not result:
        err = ctypes.get_last_error()
        if err:
            raise ctypes.WinError(err)
    return args

@lazyobject
def GetStdHandle():
    if False:
        return 10
    return lazyimps._winapi.GetStdHandle

@lazyobject
def STDHANDLES():
    if False:
        for i in range(10):
            print('nop')
    'Tuple of the Windows handles for (stdin, stdout, stderr).'
    hs = [lazyimps._winapi.STD_INPUT_HANDLE, lazyimps._winapi.STD_OUTPUT_HANDLE, lazyimps._winapi.STD_ERROR_HANDLE]
    hcons = []
    for h in hs:
        hcon = GetStdHandle(int(h))
        hcons.append(hcon)
    return tuple(hcons)

@lazyobject
def GetConsoleMode():
    if False:
        while True:
            i = 10
    gcm = ctypes.windll.kernel32.GetConsoleMode
    gcm.errcheck = check_zero
    gcm.argtypes = (HANDLE, LPDWORD)
    return gcm

def get_console_mode(fd=1):
    if False:
        i = 10
        return i + 15
    "Get the mode of the active console input, output, or error\n    buffer. Note that if the process isn't attached to a\n    console, this function raises an EBADF IOError.\n\n    Parameters\n    ----------\n    fd : int\n        Standard buffer file descriptor, 0 for stdin, 1 for stdout (default),\n        and 2 for stderr\n    "
    mode = DWORD()
    hcon = STDHANDLES[fd]
    GetConsoleMode(hcon, byref(mode))
    return mode.value

@lazyobject
def SetConsoleMode():
    if False:
        i = 10
        return i + 15
    scm = ctypes.windll.kernel32.SetConsoleMode
    scm.errcheck = check_zero
    scm.argtypes = (HANDLE, DWORD)
    return scm

def set_console_mode(mode, fd=1):
    if False:
        for i in range(10):
            print('nop')
    "Set the mode of the active console input, output, or\n    error buffer. Note that if the process isn't attached to a\n    console, this function raises an EBADF IOError.\n\n    Parameters\n    ----------\n    mode : int\n        Mode flags to set on the handle.\n    fd : int, optional\n        Standard buffer file descriptor, 0 for stdin, 1 for stdout (default),\n        and 2 for stderr\n    "
    hcon = STDHANDLES[fd]
    SetConsoleMode(hcon, mode)

def enable_virtual_terminal_processing():
    if False:
        return 10
    'Enables virtual terminal processing on Windows.\n    This includes ANSI escape sequence interpretation.\n    See http://stackoverflow.com/a/36760881/2312428\n    '
    SetConsoleMode(GetStdHandle(-11), 7)

@lazyobject
def COORD():
    if False:
        i = 10
        return i + 15
    if platform.has_prompt_toolkit():
        import prompt_toolkit.win32_types
        return prompt_toolkit.win32_types.COORD

    class _COORD(ctypes.Structure):
        """Struct from the winapi, representing coordinates in the console.

        Attributes
        ----------
        X : int
            Column position
        Y : int
            Row position
        """
        _fields_ = [('X', SHORT), ('Y', SHORT)]
    return _COORD

@lazyobject
def ReadConsoleOutputCharacterA():
    if False:
        return 10
    rcoc = ctypes.windll.kernel32.ReadConsoleOutputCharacterA
    rcoc.errcheck = check_zero
    rcoc.argtypes = (HANDLE, LPCSTR, DWORD, COORD, LPDWORD)
    rcoc.restype = BOOL
    return rcoc

@lazyobject
def ReadConsoleOutputCharacterW():
    if False:
        print('Hello World!')
    rcoc = ctypes.windll.kernel32.ReadConsoleOutputCharacterW
    rcoc.errcheck = check_zero
    rcoc.argtypes = (HANDLE, LPCWSTR, DWORD, COORD, LPDWORD)
    rcoc.restype = BOOL
    return rcoc

def read_console_output_character(x=0, y=0, fd=1, buf=None, bufsize=1024, raw=False):
    if False:
        for i in range(10):
            print('nop')
    'Reads characters from the console buffer.\n\n    Parameters\n    ----------\n    x : int, optional\n        Starting column.\n    y : int, optional\n        Starting row.\n    fd : int, optional\n        Standard buffer file descriptor, 0 for stdin, 1 for stdout (default),\n        and 2 for stderr.\n    buf : ctypes.c_wchar_p if raw else ctypes.c_wchar_p, optional\n        An existing buffer to (re-)use.\n    bufsize : int, optional\n        The maximum read size.\n    raw : bool, optional\n        Whether to read in and return as bytes (True) or as a\n        unicode string (False, default).\n\n    Returns\n    -------\n    value : str\n        Result of what was read, may be shorter than bufsize.\n    '
    hcon = STDHANDLES[fd]
    if buf is None:
        if raw:
            buf = ctypes.c_char_p(b' ' * bufsize)
        else:
            buf = ctypes.c_wchar_p(' ' * bufsize)
    coord = COORD(x, y)
    n = DWORD()
    if raw:
        ReadConsoleOutputCharacterA(hcon, buf, bufsize, coord, byref(n))
    else:
        ReadConsoleOutputCharacterW(hcon, buf, bufsize, coord, byref(n))
    return buf.value[:n.value]

def pread_console(fd, buffersize, offset, buf=None):
    if False:
        while True:
            i = 10
    'This is a console-based implementation of os.pread() for windows.\n    that uses read_console_output_character().\n    '
    (cols, rows) = os.get_terminal_size(fd=fd)
    x = offset % cols
    y = offset // cols
    return read_console_output_character(x=x, y=y, fd=fd, buf=buf, bufsize=buffersize, raw=True)

@lazyobject
def CONSOLE_SCREEN_BUFFER_INFO():
    if False:
        for i in range(10):
            print('nop')
    if platform.has_prompt_toolkit():
        import prompt_toolkit.win32_types
        return prompt_toolkit.win32_types.CONSOLE_SCREEN_BUFFER_INFO
    COORD()

    class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
        """Struct from in wincon.h. See Windows API docs
        for more details.

        Attributes
        ----------
        dwSize : COORD
            Size of
        dwCursorPosition : COORD
            Current cursor location.
        wAttributes : WORD
            Flags for screen buffer.
        srWindow : SMALL_RECT
            Actual size of screen
        dwMaximumWindowSize : COORD
            Maximum window scrollback size.
        """
        _fields_ = [('dwSize', COORD), ('dwCursorPosition', COORD), ('wAttributes', WORD), ('srWindow', SMALL_RECT), ('dwMaximumWindowSize', COORD)]
    return _CONSOLE_SCREEN_BUFFER_INFO

@lazyobject
def GetConsoleScreenBufferInfo():
    if False:
        print('Hello World!')
    'Returns the windows version of the get screen buffer.'
    gcsbi = ctypes.windll.kernel32.GetConsoleScreenBufferInfo
    gcsbi.errcheck = check_zero
    gcsbi.argtypes = (HANDLE, POINTER(CONSOLE_SCREEN_BUFFER_INFO))
    gcsbi.restype = BOOL
    return gcsbi

def get_console_screen_buffer_info(fd=1):
    if False:
        print('Hello World!')
    'Returns an screen buffer info object for the relevant stdbuf.\n\n    Parameters\n    ----------\n    fd : int, optional\n        Standard buffer file descriptor, 0 for stdin, 1 for stdout (default),\n        and 2 for stderr.\n\n    Returns\n    -------\n    csbi : CONSOLE_SCREEN_BUFFER_INFO\n        Information about the console screen buffer.\n    '
    hcon = STDHANDLES[fd]
    csbi = CONSOLE_SCREEN_BUFFER_INFO()
    GetConsoleScreenBufferInfo(hcon, byref(csbi))
    return csbi

def get_cursor_position(fd=1):
    if False:
        print('Hello World!')
    'Gets the current cursor position as an (x, y) tuple.'
    csbi = get_console_screen_buffer_info(fd=fd)
    coord = csbi.dwCursorPosition
    return (coord.X, coord.Y)

def get_cursor_offset(fd=1):
    if False:
        print('Hello World!')
    'Gets the current cursor position as a total offset value.'
    csbi = get_console_screen_buffer_info(fd=fd)
    pos = csbi.dwCursorPosition
    size = csbi.dwSize
    return pos.Y * size.X + pos.X

def get_position_size(fd=1):
    if False:
        print('Hello World!')
    'Gets the current cursor position and screen size tuple:\n    (x, y, columns, lines).\n    '
    info = get_console_screen_buffer_info(fd)
    return (info.dwCursorPosition.X, info.dwCursorPosition.Y, info.dwSize.X, info.dwSize.Y)

@lazyobject
def SetConsoleScreenBufferSize():
    if False:
        return 10
    'Set screen buffer dimensions.'
    scsbs = ctypes.windll.kernel32.SetConsoleScreenBufferSize
    scsbs.errcheck = check_zero
    scsbs.argtypes = (HANDLE, COORD)
    scsbs.restype = BOOL
    return scsbs

def set_console_screen_buffer_size(x, y, fd=1):
    if False:
        i = 10
        return i + 15
    'Sets the console size for a standard buffer.\n\n    Parameters\n    ----------\n    x : int\n        Number of columns.\n    y : int\n        Number of rows.\n    fd : int, optional\n        Standard buffer file descriptor, 0 for stdin, 1 for stdout (default),\n        and 2 for stderr.\n    '
    coord = COORD()
    coord.X = x
    coord.Y = y
    hcon = STDHANDLES[fd]
    rtn = SetConsoleScreenBufferSize(hcon, coord)
    return rtn

@lazyobject
def SetConsoleCursorPosition():
    if False:
        return 10
    'Set cursor position in console.'
    sccp = ctypes.windll.kernel32.SetConsoleCursorPosition
    sccp.errcheck = check_zero
    sccp.argtypes = (HANDLE, COORD)
    sccp.restype = BOOL
    return sccp

def set_console_cursor_position(x, y, fd=1):
    if False:
        return 10
    'Sets the console cursor position for a standard buffer.\n\n    Parameters\n    ----------\n    x : int\n        Number of columns.\n    y : int\n        Number of rows.\n    fd : int, optional\n        Standard buffer file descriptor, 0 for stdin, 1 for stdout (default),\n        and 2 for stderr.\n    '
    coord = COORD()
    coord.X = x
    coord.Y = y
    hcon = STDHANDLES[fd]
    rtn = SetConsoleCursorPosition(hcon, coord)
    return rtn