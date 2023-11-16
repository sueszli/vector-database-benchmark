try:
    import framebuf, sys
except ImportError:
    print('SKIP')
    raise SystemExit
if sys.byteorder != 'little':
    print('SKIP')
    raise SystemExit

def printbuf():
    if False:
        while True:
            i = 10
    print('--8<--')
    for y in range(h):
        print(buf[y * w * 2:(y + 1) * w * 2])
    print('-->8--')
w = 4
h = 5
buf = bytearray(w * h * 2)
fbuf = framebuf.FrameBuffer(buf, w, h, framebuf.RGB565)
fbuf.fill(65535)
printbuf()
fbuf.fill(0)
printbuf()
fbuf.pixel(0, 0, 61166)
fbuf.pixel(3, 0, 60928)
fbuf.pixel(0, 4, 238)
fbuf.pixel(3, 4, 3808)
printbuf()
print(fbuf.pixel(0, 4), fbuf.pixel(1, 1))
fbuf.fill(0)
fbuf.pixel(2, 2, 65535)
printbuf()
fbuf.scroll(0, 1)
printbuf()
fbuf.scroll(1, 0)
printbuf()
fbuf.scroll(-1, -2)
printbuf()
w2 = 2
h2 = 3
buf2 = bytearray(w2 * h2 * 2)
fbuf2 = framebuf.FrameBuffer(buf2, w2, h2, framebuf.RGB565)
fbuf2.fill(0)
fbuf2.pixel(0, 0, 3808)
fbuf2.pixel(0, 2, 60928)
fbuf2.pixel(1, 0, 238)
fbuf2.pixel(1, 2, 57358)
fbuf.fill(65535)
fbuf.blit(fbuf2, 3, 3, 0)
fbuf.blit(fbuf2, -1, -1, 0)
fbuf.blit(fbuf2, 16, 16, 0)
printbuf()