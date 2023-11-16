from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
import random

def find_focus(pattern):
    if False:
        for i in range(10):
            print('nop')
    '\n    Trues to find the focus with the shortest alt text containing `pattern`.\n    If found, returns a random coordinate within that displayable.\n\n    If `pattern` is None, returns a random coordinate that will trigger the\n    default focus.\n\n    If `pattern` could not be found, returns None, None.\n    '

    def match(f):
        if False:
            i = 10
            return i + 15
        if pattern is None:
            if f.x is None:
                return 'default'
            else:
                return None
        if f.x is None:
            t = renpy.display.tts.root._tts_all()
        else:
            t = f.widget._tts_all()
        if pattern.lower() in t.lower():
            return t
        else:
            return None
    matching = []
    for f in renpy.display.focus.focus_list:
        alt = match(f)
        if alt is not None:
            matching.append((alt, f))
    if not matching:
        return None
    matching.sort(key=lambda a: (len(a[0]), a[0]))
    return matching[0][1]

def relative_position(x, posx, width):
    if False:
        for i in range(10):
            print('nop')
    if posx is not None:
        if isinstance(posx, float):
            x = int(posx * (width - 1))
        else:
            x = posx
    return int(x)

def find_position(f, position):
    if False:
        print('Hello World!')
    '\n    Returns the virtual position of a coordinate located within focus `f`.\n    If position is (None, None) returns the current mouse position (if in\n    the focus), or a random position.\n\n    If `f` is None, returns a position relative to the screen as a whole.\n    '
    (posx, posy) = position
    if renpy.test.testmouse.mouse_pos is not None:
        (x, y) = renpy.test.testmouse.mouse_pos
    else:
        x = random.randrange(renpy.config.screen_width)
        y = random.randrange(renpy.config.screen_height)
    if f is None:
        return (relative_position(x, posx, renpy.config.screen_width), relative_position(y, posy, renpy.config.screen_height))
    orig_f = f
    if f.x is None:
        f = f.copy()
        f.x = 0
        f.y = 0
        f.w = renpy.config.screen_width
        f.h = renpy.config.screen_height
    x = relative_position(x, posx, f.w) + f.x
    y = relative_position(y, posy, f.h) + f.y
    for _i in range(100):
        x = int(x)
        y = int(y)
        nf = renpy.display.render.focus_at_point(x, y)
        if nf is None:
            if orig_f.x is None:
                return (x, y)
        elif nf.widget == f.widget and nf.arg == f.arg:
            return (x, y)
        x = random.randrange(f.x, f.x + f.w)
        y = random.randrange(f.y, f.y + f.h)
    else:
        print()
        raise Exception('Could not locate the displayable.')