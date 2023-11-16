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
    bytes_per_row = w // 2
    for y in range(h):
        for x in range(bytes_per_row):
            print('%02x' % buf[x + y * bytes_per_row], end='')
        print()
    print('-->8--')
w = 10
h = 10
buf = bytearray(w * h // 2)
fbuf = framebuf.FrameBuffer(buf, w, h, framebuf.GS4_HMSB)

def prepare_buffer():
    if False:
        return 10
    fbuf.fill(0)
    fbuf.rect(2, 0, 6, 10, 7, True)
    fbuf.rect(0, 2, 10, 6, 1, True)
prepare_buffer()
printbuf()
fbuf.scroll(5, -1)
printbuf()
prepare_buffer()
fbuf.scroll(-5, 5)
printbuf()
prepare_buffer()
fbuf.scroll(15, 7)
fbuf.scroll(10, -1)
fbuf.scroll(1, -10)
printbuf()