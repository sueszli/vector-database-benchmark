import numba as nb

@nb.jit(nopython=True, parallel=True)
def foo():
    if False:
        for i in range(10):
            print('nop')
    pass