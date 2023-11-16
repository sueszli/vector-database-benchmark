def f(x: cython.double):
    if False:
        while True:
            i = 10
    return x ** 2 - x

def integrate_f(a: cython.double, b: cython.double, N: cython.int):
    if False:
        while True:
            i = 10
    i: cython.int
    s: cython.double
    dx: cython.double
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx