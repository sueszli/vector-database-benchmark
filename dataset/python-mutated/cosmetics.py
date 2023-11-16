from __future__ import print_function
import os
import platform
is_win = platform.system() == 'Windows'

def getTerminalSize():
    if False:
        i = 10
        return i + 15
    'Return the size of the terminal : COLUMNS, LINES'
    env = os.environ

    def ioctl_GWINSZ(fd):
        if False:
            return 10
        try:
            import fcntl
            import termios
            import struct
            import os
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
    return (int(cr[1]), int(cr[0]))
WIDTH = getTerminalSize()[0]
colors = {'red': '\x1b[91;1m', 'end': '\x1b[0m', 'green': '\x1b[92;1m', 'lightcyan': '\x1b[96m', 'blue': '\x1b[94;1m'}
if is_win:
    colors = {'red': '', 'end': '', 'green': '', 'lightcyan': '', 'blue': ''}

def write_colored(text, color, already_printed=0):
    if False:
        i = 10
        return i + 15
    text_colored = colors[color] + text + colors['end']
    print(' ' * (WIDTH - already_printed - len(text)) + text_colored)

def write_underline(text):
    if False:
        while True:
            i = 10
    print('\x1b[4m' + text + colors['end'])