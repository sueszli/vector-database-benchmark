"""ctypes hackery to enable color processing on windows.

See: https://github.com/pre-commit/pre-commit/blob/cb40e96/pre_commit/color.py
"""
from __future__ import annotations
import sys
if sys.platform == 'win32':

    def _enable() -> None:
        if False:
            for i in range(10):
                print('nop')
        from ctypes import POINTER
        from ctypes import windll
        from ctypes import WinError
        from ctypes import WINFUNCTYPE
        from ctypes.wintypes import BOOL
        from ctypes.wintypes import DWORD
        from ctypes.wintypes import HANDLE
        STD_ERROR_HANDLE = -12
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 4

        def bool_errcheck(result, func, args):
            if False:
                return 10
            if not result:
                raise WinError()
            return args
        GetStdHandle = WINFUNCTYPE(HANDLE, DWORD)(('GetStdHandle', windll.kernel32), ((1, 'nStdHandle'),))
        GetConsoleMode = WINFUNCTYPE(BOOL, HANDLE, POINTER(DWORD))(('GetConsoleMode', windll.kernel32), ((1, 'hConsoleHandle'), (2, 'lpMode')))
        GetConsoleMode.errcheck = bool_errcheck
        SetConsoleMode = WINFUNCTYPE(BOOL, HANDLE, DWORD)(('SetConsoleMode', windll.kernel32), ((1, 'hConsoleHandle'), (1, 'dwMode')))
        SetConsoleMode.errcheck = bool_errcheck
        stderr = GetStdHandle(STD_ERROR_HANDLE)
        flags = GetConsoleMode(stderr)
        SetConsoleMode(stderr, flags | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    try:
        _enable()
    except OSError:
        terminal_supports_color = False
    else:
        terminal_supports_color = True
else:
    terminal_supports_color = True