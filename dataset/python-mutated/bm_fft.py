import math, cmath

def transform_radix2(vector, inverse):
    if False:
        for i in range(10):
            print('nop')

    def reverse(x, bits):
        if False:
            i = 10
            return i + 15
        y = 0
        for i in range(bits):
            y = y << 1 | x & 1
            x >>= 1
        return y
    n = len(vector)
    levels = int(math.log(n) / math.log(2))
    coef = (2 if inverse else -2) * cmath.pi / n
    exptable = [cmath.rect(1, i * coef) for i in range(n // 2)]
    vector = [vector[reverse(i, levels)] for i in range(n)]
    size = 2
    while size <= n:
        halfsize = size // 2
        tablestep = n // size
        for i in range(0, n, size):
            k = 0
            for j in range(i, i + halfsize):
                temp = vector[j + halfsize] * exptable[k]
                vector[j + halfsize] = vector[j] - temp
                vector[j] += temp
                k += tablestep
        size *= 2
    return vector
bm_params = {(50, 25): (2, 128), (100, 100): (3, 256), (1000, 1000): (20, 512), (5000, 1000): (100, 512)}

def bm_setup(params):
    if False:
        i = 10
        return i + 15
    state = None
    signal = [math.cos(2 * math.pi * i / params[1]) + 0j for i in range(params[1])]
    fft = None
    fft_inv = None

    def run():
        if False:
            return 10
        nonlocal fft, fft_inv
        for _ in range(params[0]):
            fft = transform_radix2(signal, False)
            fft_inv = transform_radix2(fft, True)

    def result():
        if False:
            while True:
                i = 10
        nonlocal fft, fft_inv
        fft[1] -= 0.5 * params[1]
        fft[-1] -= 0.5 * params[1]
        fft_ok = all((abs(f) < 0.001 for f in fft))
        for i in range(len(fft_inv)):
            fft_inv[i] -= params[1] * signal[i]
        fft_inv_ok = all((abs(f) < 0.001 for f in fft_inv))
        return (params[0] * params[1], (fft_ok, fft_inv_ok))
    return (run, result)