try:
    import framebuf, sys
except ImportError:
    print('SKIP')
    raise SystemExit
if sys.byteorder != 'little':
    print('SKIP')
    raise SystemExit

class FB(framebuf.FrameBuffer):

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        self.n = n
        super().__init__(bytearray(2 * n * n), n, n, framebuf.RGB565)

    def foo(self):
        if False:
            return 10
        self.hline(0, 2, self.n, 772)
fb = FB(n=3)
fb.pixel(0, 0, 258)
fb.foo()
print(bytes(fb))
fb2 = framebuf.FrameBuffer(bytearray(2 * 3 * 3), 3, 3, framebuf.RGB565)
fb.fill(0)
fb.pixel(0, 0, 1286)
fb.pixel(2, 2, 1800)
fb2.blit(fb, 0, 0)
print(bytes(fb2))

class NotAFrameBuf:
    pass
try:
    fb.blit(NotAFrameBuf(), 0, 0)
except TypeError:
    print('TypeError')
try:
    fb.blit(None, 0, 0)
except TypeError:
    print('TypeError')