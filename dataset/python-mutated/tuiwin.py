from visidata import VisiData, vd
vd._parentscrs = {}

@VisiData.api
def subwindow(vd, scr, x, y, w, h):
    if False:
        for i in range(10):
            print('nop')
    'Return subwindow with its (0,0) at (x,y) relative to parent scr.  Replacement for scr.derwin() to track parent scr.'
    newscr = scr.derwin(h, w, y, x)
    vd._parentscrs[newscr] = scr
    return newscr

@VisiData.api
def getrootxy(vd, scr):
    if False:
        while True:
            i = 10
    (px, py) = (0, 0)
    while scr in vd._parentscrs:
        (dy, dx) = scr.getparyx()
        if dy > 0:
            py += dy
        if dx > 0:
            px += dx
        scr = vd._parentscrs[scr]
    return (px, py)