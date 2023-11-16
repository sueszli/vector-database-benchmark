try:
    import framebuf
except ImportError:
    print('SKIP')
    raise SystemExit

def printbuf():
    if False:
        while True:
            i = 10
    print('--8<--')
    for y in range(h):
        print(buf[y * w // 2:(y + 1) * w // 2])
    print('-->8--')
w = 16
h = 8
buf = bytearray(w * h // 2)
fbuf = framebuf.FrameBuffer(buf, w, h, framebuf.GS4_HMSB)
fbuf.fill(15)
printbuf()
fbuf.fill(160)
printbuf()
fbuf.pixel(0, 0, 1)
printbuf()
fbuf.pixel(w - 1, 0, 2)
printbuf()
fbuf.pixel(w - 1, h - 1, 3)
printbuf()
fbuf.pixel(0, h - 1, 4)
printbuf()
print(fbuf.pixel(0, 0), fbuf.pixel(w - 1, 0), fbuf.pixel(w - 1, h - 1), fbuf.pixel(0, h - 1))
print(fbuf.pixel(1, 0), fbuf.pixel(w - 2, 0), fbuf.pixel(w - 2, h - 1), fbuf.pixel(1, h - 1))
fbuf.fill_rect(0, 0, w, h, 15)
printbuf()
fbuf.fill_rect(0, 0, w, h, 240)
fbuf.fill_rect(1, 0, w // 2 + 1, 1, 241)
printbuf()
fbuf.fill_rect(1, 0, w // 2 + 1, 1, 16)
fbuf.fill_rect(1, 0, w // 2, 1, 241)
printbuf()
fbuf.fill_rect(1, 0, w // 2, 1, 16)
fbuf.fill_rect(0, h - 4, w // 2 + 1, 4, 175)
printbuf()
fbuf.fill_rect(0, h - 4, w // 2 + 1, 4, 176)
fbuf.fill_rect(0, h - 4, w // 2, 4, 175)
printbuf()
fbuf.fill_rect(0, h - 4, w // 2, 4, 176)