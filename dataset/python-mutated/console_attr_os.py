"""OS specific console_attr helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from fire.console import encoding

def GetTermSize():
    if False:
        i = 10
        return i + 15
    'Gets the terminal x and y dimensions in characters.\n\n  _GetTermSize*() helper functions taken from:\n    http://stackoverflow.com/questions/263890/\n\n  Returns:\n    (columns, lines): A tuple containing the terminal x and y dimensions.\n  '
    xy = None
    for get_terminal_size in (_GetTermSizePosix, _GetTermSizeWindows, _GetTermSizeEnvironment, _GetTermSizeTput):
        try:
            xy = get_terminal_size()
            if xy:
                break
        except:
            pass
    return xy or (80, 24)

def _GetTermSizePosix():
    if False:
        return 10
    'Returns the Posix terminal x and y dimensions.'
    import fcntl
    import struct
    import termios

    def _GetXY(fd):
        if False:
            i = 10
            return i + 15
        'Returns the terminal (x,y) size for fd.\n\n    Args:\n      fd: The terminal file descriptor.\n\n    Returns:\n      The terminal (x,y) size for fd or None on error.\n    '
        try:
            rc = struct.unpack(b'hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, 'junk'))
            return (rc[1], rc[0]) if rc else None
        except:
            return None
    xy = _GetXY(0) or _GetXY(1) or _GetXY(2)
    if not xy:
        fd = None
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            xy = _GetXY(fd)
        except:
            xy = None
        finally:
            if fd is not None:
                os.close(fd)
    return xy

def _GetTermSizeWindows():
    if False:
        for i in range(10):
            print('nop')
    'Returns the Windows terminal x and y dimensions.'
    import struct
    from ctypes import create_string_buffer
    from ctypes import windll
    h = windll.kernel32.GetStdHandle(-12)
    csbi = create_string_buffer(22)
    if not windll.kernel32.GetConsoleScreenBufferInfo(h, csbi):
        return None
    (unused_bufx, unused_bufy, unused_curx, unused_cury, unused_wattr, left, top, right, bottom, unused_maxx, unused_maxy) = struct.unpack(b'hhhhHhhhhhh', csbi.raw)
    x = right - left + 1
    y = bottom - top + 1
    return (x, y)

def _GetTermSizeEnvironment():
    if False:
        return 10
    'Returns the terminal x and y dimensions from the environment.'
    return (int(os.environ['COLUMNS']), int(os.environ['LINES']))

def _GetTermSizeTput():
    if False:
        for i in range(10):
            print('nop')
    'Returns the terminal x and y dimensions from tput(1).'
    import subprocess
    output = encoding.Decode(subprocess.check_output(['tput', 'cols'], stderr=subprocess.STDOUT))
    cols = int(output)
    output = encoding.Decode(subprocess.check_output(['tput', 'lines'], stderr=subprocess.STDOUT))
    rows = int(output)
    return (cols, rows)
_ANSI_CSI = '\x1b'
_CONTROL_D = '\x04'
_CONTROL_Z = '\x1a'
_WINDOWS_CSI_1 = '\x00'
_WINDOWS_CSI_2 = 'à'

def GetRawKeyFunction():
    if False:
        i = 10
        return i + 15
    'Returns a function that reads one keypress from stdin with no echo.\n\n  Returns:\n    A function that reads one keypress from stdin with no echo or a function\n    that always returns None if stdin does not support it.\n  '
    for get_raw_key_function in (_GetRawKeyFunctionPosix, _GetRawKeyFunctionWindows):
        try:
            return get_raw_key_function()
        except:
            pass
    return lambda : None

def _GetRawKeyFunctionPosix():
    if False:
        i = 10
        return i + 15
    '_GetRawKeyFunction helper using Posix APIs.'
    import tty
    import termios

    def _GetRawKeyPosix():
        if False:
            for i in range(10):
                print('nop')
        'Reads and returns one keypress from stdin, no echo, using Posix APIs.\n\n    Returns:\n      The key name, None for EOF, <*> for function keys, otherwise a\n      character.\n    '
        ansi_to_key = {'A': '<UP-ARROW>', 'B': '<DOWN-ARROW>', 'D': '<LEFT-ARROW>', 'C': '<RIGHT-ARROW>', '5': '<PAGE-UP>', '6': '<PAGE-DOWN>', 'H': '<HOME>', 'F': '<END>', 'M': '<DOWN-ARROW>', 'S': '<PAGE-UP>', 'T': '<PAGE-DOWN>'}
        sys.stdout.flush()
        fd = sys.stdin.fileno()

        def _GetKeyChar():
            if False:
                while True:
                    i = 10
            return encoding.Decode(os.read(fd, 1))
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            c = _GetKeyChar()
            if c == _ANSI_CSI:
                c = _GetKeyChar()
                while True:
                    if c == _ANSI_CSI:
                        return c
                    if c.isalpha():
                        break
                    prev_c = c
                    c = _GetKeyChar()
                    if c == '~':
                        c = prev_c
                        break
                return ansi_to_key.get(c, '')
        except:
            c = None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return None if c in (_CONTROL_D, _CONTROL_Z) else c
    return _GetRawKeyPosix

def _GetRawKeyFunctionWindows():
    if False:
        for i in range(10):
            print('nop')
    '_GetRawKeyFunction helper using Windows APIs.'
    import msvcrt

    def _GetRawKeyWindows():
        if False:
            i = 10
            return i + 15
        'Reads and returns one keypress from stdin, no echo, using Windows APIs.\n\n    Returns:\n      The key name, None for EOF, <*> for function keys, otherwise a\n      character.\n    '
        windows_to_key = {'H': '<UP-ARROW>', 'P': '<DOWN-ARROW>', 'K': '<LEFT-ARROW>', 'M': '<RIGHT-ARROW>', 'I': '<PAGE-UP>', 'Q': '<PAGE-DOWN>', 'G': '<HOME>', 'O': '<END>'}
        sys.stdout.flush()

        def _GetKeyChar():
            if False:
                while True:
                    i = 10
            return encoding.Decode(msvcrt.getch())
        c = _GetKeyChar()
        if c in (_WINDOWS_CSI_1, _WINDOWS_CSI_2):
            return windows_to_key.get(_GetKeyChar(), '')
        return None if c in (_CONTROL_D, _CONTROL_Z) else c
    return _GetRawKeyWindows