from __future__ import print_function
import renpy

def none_is_null(o):
    if False:
        i = 10
        return i + 15
    if o is None:
        return renpy.display.layout.Null()
    else:
        return renpy.easy.displayable(o)

def none_is_0(o):
    if False:
        print('Hello World!')
    if o is None:
        return 0
    else:
        return o

def expand_focus_mask(v):
    if False:
        return 10
    if v is None:
        return v
    elif v is False:
        return v
    elif v is True:
        return v
    elif callable(v):
        return v
    else:
        return renpy.easy.displayable(v)

def expand_outlines(l):
    if False:
        for i in range(10):
            print('nop')
    rv = []
    for i in l:
        if len(i) == 2:
            rv.append((i[0], renpy.easy.color(i[1]), 0, 0))
        else:
            rv.append((i[0], renpy.easy.color(i[1]), i[2], i[3]))
    return rv
ANCHORS = dict(left=0.0, right=1.0, center=0.5, top=0.0, bottom=1.0)

def expand_anchor(v):
    if False:
        for i in range(10):
            print('nop')
    '\n    Turns an anchor into a number.\n    '
    try:
        return ANCHORS.get(v, v)
    except Exception:
        for n in ANCHORS:
            o = getattr(renpy.store, n, None)
            if o is None:
                continue
            if v is o:
                return ANCHORS[n]
        raise