import numpy
from cython.cimports.numpy import int32_t

def main():
    if False:
        i = 10
        return i + 15
    a: int32_t[:] = numpy.arange(10, dtype=numpy.int32)
    a = a[::2]
    print(a)
    print(numpy.asarray(a))
    print(a.base)