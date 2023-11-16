from __future__ import absolute_import
import os
import sys

class FG(object):
    """Unix terminal foreground color codes (16-color)."""
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    YELLOW = '\x1b[33m'
    BLUE = '\x1b[34m'
    MAGENTA = '\x1b[35m'
    CYAN = '\x1b[36m'
    WHITE = '\x1b[37m'
    BOLD_RED = '\x1b[1;31m'
    BOLD_GREEN = '\x1b[1;32m'
    BOLD_YELLOW = '\x1b[1;33m'
    BOLD_BLUE = '\x1b[1;34m'
    BOLD_MAGENTA = '\x1b[1;35m'
    BOLD_CYAN = '\x1b[1;36m'
    BOLD_WHITE = '\x1b[1;37m'
    NONE = '\x1b[0m'

class BG(object):
    """Unix terminal background color codes (16-color)."""
    BLACK = '\x1b[40m'
    RED = '\x1b[41m'
    GREEN = '\x1b[42m'
    YELLOW = '\x1b[43m'
    BLUE = '\x1b[44m'
    MAGENTA = '\x1b[45m'
    CYAN = '\x1b[46m'
    WHITE = '\x1b[47m'
    NONE = '\x1b[0m'

def color_string(s, fg, bg=''):
    if False:
        while True:
            i = 10
    return fg + bg + s + FG.NONE

def re_color_string(compiled_pattern, s, fg):
    if False:
        while True:
            i = 10
    return compiled_pattern.sub(fg + '\\1' + FG.NONE, s)

def allow_color():
    if False:
        return 10
    if os.name != 'posix':
        return False
    if not sys.stdout.isatty():
        return False
    try:
        import curses
        curses.setupterm()
        return curses.tigetnum('colors') > 2
    except curses.error:
        return False