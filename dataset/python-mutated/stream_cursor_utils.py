"""
Stream cursor utilities for moving cursor in the terminal.
"""
import os
import platform
ESC = '\x1b['
if platform.system().lower() == 'windows':
    try:
        os.system('color')
    except Exception:
        pass

class CursorFormatter:
    """
    Base class for defining how cursor is to be manipulated.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def cursor_format(self, count):
        if False:
            return 10
        pass

class CursorUpFormatter(CursorFormatter):
    """
    Class for formatting and outputting moving the cursor up within the stream of bytes.
    """

    def cursor_format(self, count=0):
        if False:
            while True:
                i = 10
        return ESC + str(count) + 'A'

class CursorDownFormatter(CursorFormatter):
    """
    Class for formatting and outputting moving the cursor down within the stream of bytes.
    """

    def cursor_format(self, count=0):
        if False:
            return 10
        return ESC + str(count) + 'B'

class ClearLineFormatter(CursorFormatter):
    """
    Class for formatting and outputting clearing the cursor within the stream of bytes.
    """

    def cursor_format(self, count=0):
        if False:
            for i in range(10):
                print('nop')
        return ESC + str(count) + 'K'

class CursorLeftFormatter(CursorFormatter):
    """
    Class for formatting and outputting moving the cursor left within the stream of bytes.
    """

    def cursor_format(self, count=0):
        if False:
            return 10
        return ESC + str(count) + 'G'