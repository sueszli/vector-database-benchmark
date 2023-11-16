try:
    import framebuf
except ImportError:
    print('SKIP')
    raise SystemExit

def printbuf():
    if False:
        i = 10
        return i + 15
    print('--8<--')
    for y in range(h):
        for x in range(w):
            print('%02x' % buf[x + y * w], end='')
        print()
    print('-->8--')
w = 30
h = 30
buf = bytearray(w * h)
fbuf = framebuf.FrameBuffer(buf, w, h, framebuf.GS8)
fbuf.fill(0)
fbuf.ellipse(15, 15, 12, 6, 255, False)
printbuf()
fbuf.fill(0)
fbuf.ellipse(15, 15, 6, 12, 170, True)
printbuf()
for m in (0, 1, 2, 4, 8, 10):
    fbuf.fill(0)
    fbuf.ellipse(15, 15, 6, 12, 170, False, m)
    printbuf()
    fbuf.fill(0)
    fbuf.ellipse(15, 15, 6, 12, 170, True, m)
    printbuf()
for (x, y) in ((4, 4), (26, 4), (26, 26), (4, 26)):
    fbuf.fill(0)
    fbuf.ellipse(x, y, 6, 12, 170, False)
    printbuf()
    fbuf.fill(0)
    fbuf.ellipse(x, y, 6, 12, 170, True)
    printbuf()