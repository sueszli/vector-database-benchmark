from cython.cimports.libc import math
from cython.cimports.libc.math import ceil

def libc_math_ceil(x):
    if False:
        print('Hello World!')
    '\n    >>> libc_math_ceil(1.5)\n    [2, 2]\n    '
    return [int(n) for n in [ceil(x), math.ceil(x)]]