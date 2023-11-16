from __future__ import annotations
import cython
try:
    import typing
except ImportError:
    pass
import numpy
COMPILED = cython.compiled

def one_dim(a: cython.double[:]):
    if False:
        print('Hello World!')
    '\n    >>> a = numpy.ones((10,), numpy.double)\n    >>> one_dim(a)\n    (2.0, 1)\n    '
    a[0] *= 2
    return (a[0], a.ndim)

def one_dim_ccontig(a: cython.double[::1]):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> a = numpy.ones((10,), numpy.double)\n    >>> one_dim_ccontig(a)\n    (2.0, 1)\n    '
    a[0] *= 2
    return (a[0], a.ndim)

def two_dim(a: cython.double[:, :]):
    if False:
        return 10
    '\n    >>> a = numpy.ones((10, 10), numpy.double)\n    >>> two_dim(a)\n    (3.0, 1.0, 2)\n    '
    a[0, 0] *= 3
    return (a[0, 0], a[0, 1], a.ndim)

def variable_annotation(a):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> a = numpy.ones((10,), numpy.double)\n    >>> variable_annotation(a)\n    2.0\n    '
    b: cython.double[:]
    b = None
    if cython.compiled:
        assert cython.typeof(b) == 'double[:]', cython.typeof(b)
    b = a
    b[1] += 1
    b[2] += 2
    return b[1]

def slice_none(m: cython.double[:]):
    if False:
        return 10
    '\n    >>> try:\n    ...     a = slice_none(None)\n    ... except TypeError as exc:\n    ...     assert COMPILED\n    ...     if "Argument \'m\' must not be None" not in str(exc): raise\n    ... else:\n    ...     assert a == 1\n    ...     assert not COMPILED\n    '
    return 1 if m is None else 2

def slice_optional(m: typing.Optional[cython.double[:]]):
    if False:
        print('Hello World!')
    "\n    >>> slice_optional(None)\n    1\n    >>> a = numpy.ones((10,), numpy.double)\n    >>> slice_optional(a)\n    2\n\n    # Make sure that we actually evaluate the type and don't just accept everything.\n    >>> try:\n    ...     x = slice_optional(123)\n    ... except TypeError as exc:\n    ...     if not COMPILED: raise\n    ... else:\n    ...     assert not COMPILED\n    "
    return 1 if m is None else 2

@cython.nogil
@cython.cfunc
def _one_dim_nogil_cfunc(a: cython.double[:]) -> cython.double:
    if False:
        print('Hello World!')
    a[0] *= 2
    return a[0]

def one_dim_nogil_cfunc(a: cython.double[:]):
    if False:
        print('Hello World!')
    '\n    >>> a = numpy.ones((10,), numpy.double)\n    >>> one_dim_nogil_cfunc(a)\n    2.0\n    '
    with cython.nogil:
        result = _one_dim_nogil_cfunc(a)
    return result

def generic_object_memoryview(a: object[:]):
    if False:
        return 10
    '\n    >>> a = numpy.ones((10,), dtype=object)\n    >>> generic_object_memoryview(a)\n    10\n    '
    sum = 0
    for ai in a:
        sum += ai
    if cython.compiled:
        assert cython.typeof(a) == 'object[:]', cython.typeof(a)
    return sum

def generic_object_memoryview_contig(a: object[::1]):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> a = numpy.ones((10,), dtype=object)\n    >>> generic_object_memoryview_contig(a)\n    10\n    '
    sum = 0
    for ai in a:
        sum += ai
    if cython.compiled:
        assert cython.typeof(a) == 'object[::1]', cython.typeof(a)
    return sum

@cython.cclass
class C:
    x: cython.int

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.x = value

def ext_type_object_memoryview(a: C[:]):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> a = numpy.array([C(i) for i in range(10)], dtype=object)\n    >>> ext_type_object_memoryview(a)\n    45\n    '
    sum = 0
    for ai in a:
        sum += ai.x
    if cython.compiled:
        assert cython.typeof(a) == 'C[:]', cython.typeof(a)
    return sum

def ext_type_object_memoryview_contig(a: C[::1]):
    if False:
        return 10
    '\n    >>> a = numpy.array([C(i) for i in range(10)], dtype=object)\n    >>> ext_type_object_memoryview_contig(a)\n    45\n    '
    sum = 0
    for ai in a:
        sum += ai.x
    if cython.compiled:
        assert cython.typeof(a) == 'C[::1]', cython.typeof(a)
    return sum