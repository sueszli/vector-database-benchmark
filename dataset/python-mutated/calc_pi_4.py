import cython

@cython.profile(False)
@cython.cfunc
@cython.inline
@cython.exceptval(-1.0)
def recip_square(i: cython.longlong) -> float:
    if False:
        while True:
            i = 10
    return 1.0 / (i * i)

def approx_pi(n: cython.int=10000000):
    if False:
        return 10
    val: cython.double = 0.0
    k: cython.int
    for k in range(1, n + 1):
        val += recip_square(k)
    return (6 * val) ** 0.5