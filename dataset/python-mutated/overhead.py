from cython.cimports.cpython import array
import array
a = cython.declare(array.array, array.array('i', [1, 2, 3]))
ca = cython.declare(cython.int[:], a)

@cython.cfunc
def overhead(a: cython.object) -> cython.int:
    if False:
        return 10
    ca: cython.int[:] = a
    return ca[0]

@cython.cfunc
def no_overhead(ca: cython.int[:]) -> cython.int:
    if False:
        return 10
    return ca[0]
print(overhead(a))
print(no_overhead(ca))