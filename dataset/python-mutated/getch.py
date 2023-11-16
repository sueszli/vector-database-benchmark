"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""

class _Getch(object):
    """
    Gets a single character from standard input.  Does not echo to
    the screen (reference: http://code.activestate.com/recipes/134892/)
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        try:
            self.impl = _GetchWindows()
        except ImportError:
            try:
                self.impl = _GetchMacCarbon()
            except (AttributeError, ImportError):
                self.impl = _GetchUnix()

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.impl()

class _GetchUnix(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        __import__('tty')

    def __call__(self):
        if False:
            return 10
        import sys
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

class _GetchWindows(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        __import__('msvcrt')

    def __call__(self):
        if False:
            i = 10
            return i + 15
        import msvcrt
        return msvcrt.getch()

class _GetchMacCarbon(object):
    """
    A function which returns the current ASCII key that is down;
    if no ASCII key is down, the null string is returned.  The
    page http://www.mactech.com/macintosh-c/chap02-1.html was
    very helpful in figuring out how to do this.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        import Carbon
        getattr(Carbon, 'Evt')

    def __call__(self):
        if False:
            print('Hello World!')
        import Carbon
        if Carbon.Evt.EventAvail(8)[0] == 0:
            return ''
        else:
            (what, msg, when, where, mod) = Carbon.Evt.GetNextEvent(8)[1]
            return chr(msg & 255)
getch = _Getch()