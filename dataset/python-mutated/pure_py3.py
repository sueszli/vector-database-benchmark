import cython
is_compiled = cython.compiled
MyUnion = cython.union(n=cython.int, x=cython.double)
MyStruct = cython.struct(is_integral=cython.bint, data=MyUnion)
MyStruct2 = cython.typedef(MyStruct[2])

@cython.annotation_typing(False)
def test_annotation_typing(x: cython.int) -> cython.int:
    if False:
        while True:
            i = 10
    '\n    >>> test_annotation_typing("Petits pains")\n    \'Petits pains\'\n    '
    return x

@cython.ccall
def test_return_type(n: cython.int) -> cython.double:
    if False:
        while True:
            i = 10
    '\n    >>> test_return_type(389)\n    389.0\n    '
    assert cython.typeof(n) == 'int', cython.typeof(n)
    return n if is_compiled else float(n)

def test_struct(n: cython.int, x: cython.double) -> MyStruct2:
    if False:
        while True:
            i = 10
    "\n    >>> test_struct(389, 1.64493)\n    (389, 1.64493)\n    >>> d = test_struct.__annotations__\n    >>> sorted(d)\n    ['n', 'return', 'x']\n    "
    assert cython.typeof(n) == 'int', cython.typeof(n)
    if is_compiled:
        assert cython.typeof(x) == 'double', cython.typeof(x)
    else:
        assert cython.typeof(x) == 'float', cython.typeof(x)
    a = cython.declare(MyStruct2)
    a[0] = MyStruct(is_integral=True, data=MyUnion(n=n))
    a[1] = MyStruct(is_integral=False, data={'x': x})
    return (a[0].data.n, a[1].data.x)

@cython.ccall
def c_call(x) -> cython.double:
    if False:
        while True:
            i = 10
    return x

def call_ccall(x):
    if False:
        print('Hello World!')
    "\n    Test that a declared return type is honoured when compiled.\n\n    >>> result, return_type = call_ccall(1)\n\n    >>> (not is_compiled and 'double') or return_type\n    'double'\n    >>> (is_compiled and 'int') or return_type\n    'int'\n\n    >>> (not is_compiled and 1.0) or result\n    1.0\n    >>> (is_compiled and 1) or result\n    1\n    "
    ret = c_call(x)
    return (ret, cython.typeof(ret))

@cython.cfunc
@cython.inline
def cdef_inline(x) -> cython.double:
    if False:
        return 10
    return x + 1

def call_cdef_inline(x):
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> result, return_type = call_cdef_inline(1)\n    >>> (not is_compiled and 'float') or type(result).__name__\n    'float'\n    >>> (not is_compiled and 'double') or return_type\n    'double'\n    >>> (is_compiled and 'int') or return_type\n    'int'\n    >>> result == 2.0  or  result\n    True\n    "
    ret = cdef_inline(x)
    return (ret, cython.typeof(ret))

@cython.cfunc
def test_cdef_return_object(x: object) -> object:
    if False:
        i = 10
        return i + 15
    '\n    Test support of python object in annotations\n    >>> test_cdef_return_object(3)\n    3\n    >>> test_cdef_return_object(None)\n    Traceback (most recent call last):\n        ...\n    RuntimeError\n    '
    if x:
        return x
    else:
        raise RuntimeError()