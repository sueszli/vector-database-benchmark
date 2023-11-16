from cython.cimports.libc.math import sin

@cython.cfunc
def f(x: cython.double) -> cython.double:
    if False:
        print('Hello World!')
    return sin(x * x)