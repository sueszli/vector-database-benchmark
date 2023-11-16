import cython
from cython.parallel import prange, parallel

def prange_regression(n: cython.int, data: list):
    if False:
        print('Hello World!')
    '\n    >>> prange_regression(10, list(range(1, 4)))\n    19\n    '
    s: cython.int = 0
    i: cython.int
    d: cython.int[3] = data
    for i in prange(n, num_threads=3, nogil=True):
        s += d[i % 3]
    return s

def prange_with_gil(n: cython.int, x):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> sum(3*i for i in range(10))\n    135\n    >>> prange_with_gil(10, 3)\n    135\n    '
    i: cython.int
    s: cython.int = 0
    for i in prange(n, num_threads=3, nogil=True):
        with cython.gil:
            s += x * i
    return s

@cython.cfunc
def use_nogil(x, i: cython.int) -> cython.int:
    if False:
        for i in range(10):
            print('nop')
    cx: cython.int = x
    with cython.nogil:
        return cx * i

def prange_with_gil_call_nogil(n: cython.int, x):
    if False:
        i = 10
        return i + 15
    '\n    >>> sum(3*i for i in range(10))\n    135\n    >>> prange_with_gil(10, 3)\n    135\n    '
    i: cython.int
    s: cython.int = 0
    for i in prange(n, num_threads=3, nogil=True):
        with cython.gil:
            s += use_nogil(x, i)
    return s