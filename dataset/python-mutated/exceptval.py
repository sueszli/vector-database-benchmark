import cython

@cython.exceptval(-1)
def func(x: cython.int) -> cython.int:
    if False:
        for i in range(10):
            print('nop')
    if x < 0:
        raise ValueError('need integer >= 0')
    return x + 1