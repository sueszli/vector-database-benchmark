@cython.cfunc
@cython.exceptval(-2, check=True)
def f(x: cython.double) -> cython.double:
    if False:
        i = 10
        return i + 15
    return x ** 2 - x