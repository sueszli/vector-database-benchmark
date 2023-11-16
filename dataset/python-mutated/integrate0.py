def f(x):
    if False:
        while True:
            i = 10
    return x ** 2 - x

def integrate_f(a, b, N):
    if False:
        while True:
            i = 10
    s = 0.0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx