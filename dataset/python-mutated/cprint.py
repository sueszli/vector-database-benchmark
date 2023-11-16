"""
Cross-platform color text printing

Based on colorama (see pyqtgraph/util/colorama/README.txt)
"""
import sys
from .colorama.win32 import windll
from .colorama.winterm import WinColor, WinStyle, WinTerm
_WIN = sys.platform.startswith('win')
if windll is not None:
    winterm = WinTerm()
else:
    _WIN = False

def winset(reset=False, fore=None, back=None, style=None, stderr=False):
    if False:
        print('Hello World!')
    if reset:
        winterm.reset_all()
    if fore is not None:
        winterm.fore(fore, stderr)
    if back is not None:
        winterm.back(back, stderr)
    if style is not None:
        winterm.style(style, stderr)
ANSI = {}
WIN = {}
for (i, color) in enumerate(['BLACK', 'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE']):
    globals()[color] = i
    globals()['BR_' + color] = i + 8
    globals()['BACK_' + color] = i + 40
    ANSI[i] = '\x1b[%dm' % (30 + i)
    ANSI[i + 8] = '\x1b[2;%dm' % (30 + i)
    ANSI[i + 40] = '\x1b[%dm' % (40 + i)
    color = 'GREY' if color == 'WHITE' else color
    WIN[i] = {'fore': getattr(WinColor, color), 'style': WinStyle.NORMAL}
    WIN[i + 8] = {'fore': getattr(WinColor, color), 'style': WinStyle.BRIGHT}
    WIN[i + 40] = {'back': getattr(WinColor, color)}
RESET = -1
ANSI[RESET] = '\x1b[0m'
WIN[RESET] = {'reset': True}

def cprint(stream, *args, **kwds):
    if False:
        while True:
            i = 10
    "\n    Print with color. Examples::\n\n        # colors are BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE\n        cprint('stdout', RED, 'This is in red. ', RESET, 'and this is normal\n')\n\n        # Adding BR_ before the color manes it bright\n        cprint('stdout', BR_GREEN, 'This is bright green.\n', RESET)\n\n        # Adding BACK_ changes background color\n        cprint('stderr', BACK_BLUE, WHITE, 'This is white-on-blue.', -1)\n\n        # Integers 0-7 for normal, 8-15 for bright, and 40-47 for background.\n        # -1 to reset.\n        cprint('stderr', 1, 'This is in red.', -1)\n\n    "
    if isinstance(stream, str):
        stream = kwds.get('stream', 'stdout')
        err = stream == 'stderr'
        stream = getattr(sys, stream)
    else:
        err = kwds.get('stderr', False)
    if hasattr(stream, 'isatty') and stream.isatty():
        if _WIN:
            for arg in args:
                if isinstance(arg, str):
                    stream.write(arg)
                else:
                    kwds = WIN[arg]
                    winset(stderr=err, **kwds)
        else:
            for arg in args:
                if isinstance(arg, str):
                    stream.write(arg)
                else:
                    stream.write(ANSI[arg])
    else:
        for arg in args:
            if isinstance(arg, str):
                stream.write(arg)

def cout(*args):
    if False:
        i = 10
        return i + 15
    "Shorthand for cprint('stdout', ...)"
    cprint('stdout', *args)

def cerr(*args):
    if False:
        i = 10
        return i + 15
    "Shorthand for cprint('stderr', ...)"
    cprint('stderr', *args)