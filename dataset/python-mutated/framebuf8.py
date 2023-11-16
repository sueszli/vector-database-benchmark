try:
    import framebuf
except ImportError:
    print('SKIP')
    raise SystemExit

def printbuf():
    if False:
        for i in range(10):
            print('nop')
    print('--8<--')
    for y in range(h):
        for x in range(w):
            print('%02x' % buf[x + y * w], end='')
        print()
    print('-->8--')
w = 8
h = 5
buf = bytearray(w * h)
fbuf = framebuf.FrameBuffer(buf, w, h, framebuf.GS8)
fbuf.fill(85)
printbuf()
fbuf.pixel(0, 0, 17)
fbuf.pixel(w - 1, 0, 34)
fbuf.pixel(0, h - 1, 51)
fbuf.pixel(w - 1, h - 1, 255)
printbuf()
print(hex(fbuf.pixel(0, h - 1)), hex(fbuf.pixel(1, 1)))