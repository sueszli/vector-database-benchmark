def mandelbrot(w, h):
    if False:
        print('Hello World!')

    def in_set(c):
        if False:
            print('Hello World!')
        z = 0
        for i in range(32):
            z = z * z + c
            if abs(z) > 100:
                return i
        return 0
    img = bytearray(w * h)
    xscale = (w - 1) / 2.4
    yscale = (h - 1) / 3.2
    for v in range(h):
        line = memoryview(img)[v * w:v * w + w]
        for u in range(w):
            c = in_set(complex(v / yscale - 2.3, u / xscale - 1.2))
            line[u] = c
    return img
bm_params = {(100, 100): (20, 20), (1000, 1000): (80, 80), (5000, 1000): (150, 150)}

def bm_setup(ps):
    if False:
        i = 10
        return i + 15
    return (lambda : mandelbrot(ps[0], ps[1]), lambda : (ps[0] * ps[1], None))