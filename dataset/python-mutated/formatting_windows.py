"""This module is used for enabling formatting on Windows."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ctypes
import os
import platform
import subprocess
import sys
try:
    import colorama
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

def initialize_or_disable():
    if False:
        print('Hello World!')
    'Enables ANSI processing on Windows or disables it as needed.'
    if HAS_COLORAMA:
        wrap = True
        if sys.stdout.isatty() and platform.release() == '10':
            wrap = False
            kernel32 = ctypes.windll.kernel32
            enable_virtual_terminal_processing = 4
            out_handle = kernel32.GetStdHandle(subprocess.STD_OUTPUT_HANDLE)
            mode = ctypes.wintypes.DWORD()
            if kernel32.GetConsoleMode(out_handle, ctypes.byref(mode)) == 0:
                wrap = True
            if not mode.value & enable_virtual_terminal_processing:
                if kernel32.SetConsoleMode(out_handle, mode.value | enable_virtual_terminal_processing) == 0:
                    wrap = True
        colorama.init(wrap=wrap)
    else:
        os.environ['ANSI_COLORS_DISABLED'] = '1'
if sys.platform.startswith('win'):
    initialize_or_disable()