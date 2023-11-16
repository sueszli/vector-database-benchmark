import cython

@cython.test_fail_if_path_exists('//MulNode')
@cython.test_assert_path_exists('//ListNode[@mult_factor]')
def cint_times_list(n: cython.int):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> cint_times_list(3)\n    []\n    [None, None, None]\n    [3, 3, 3]\n    [1, 2, 3, 1, 2, 3, 1, 2, 3]\n    '
    a = n * []
    b = n * [None]
    c = n * [n]
    d = n * [1, 2, 3]
    print(a)
    print(b)
    print(c)
    print(d)

@cython.test_fail_if_path_exists('//MulNode')
@cython.test_assert_path_exists('//ListNode[@mult_factor]')
def list_times_cint(n: cython.int):
    if False:
        return 10
    '\n    >>> list_times_cint(3)\n    []\n    [None, None, None]\n    [3, 3, 3]\n    [1, 2, 3, 1, 2, 3, 1, 2, 3]\n    '
    a = [] * n
    b = [None] * n
    c = [n] * n
    d = [1, 2, 3] * n
    print(a)
    print(b)
    print(c)
    print(d)

@cython.test_fail_if_path_exists('//MulNode')
@cython.test_assert_path_exists('//TupleNode[@mult_factor]')
def const_times_tuple(v: cython.int):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> const_times_tuple(4)\n    ()\n    (None, None)\n    (4, 4)\n    (1, 2, 3, 1, 2, 3)\n    '
    a = 2 * ()
    b = 2 * (None,)
    c = 2 * (v,)
    d = 2 * (1, 2, 3)
    print(a)
    print(b)
    print(c)
    print(d)

@cython.test_fail_if_path_exists('//MulNode')
@cython.test_assert_path_exists('//TupleNode[@mult_factor]')
def cint_times_tuple(n: cython.int):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> cint_times_tuple(3)\n    ()\n    (None, None, None)\n    (3, 3, 3)\n    (1, 2, 3, 1, 2, 3, 1, 2, 3)\n    '
    a = n * ()
    b = n * (None,)
    c = n * (n,)
    d = n * (1, 2, 3)
    print(a)
    print(b)
    print(c)
    print(d)

@cython.test_fail_if_path_exists('//MulNode')
@cython.test_assert_path_exists('//TupleNode[@mult_factor]')
def tuple_times_cint(n: cython.int):
    if False:
        while True:
            i = 10
    '\n    >>> tuple_times_cint(3)\n    ()\n    (None, None, None)\n    (3, 3, 3)\n    (1, 2, 3, 1, 2, 3, 1, 2, 3)\n    '
    a = () * n
    b = (None,) * n
    c = (n,) * n
    d = (1, 2, 3) * n
    print(a)
    print(b)
    print(c)
    print(d)

def list_times_pyint(n: cython.longlong):
    if False:
        print('Hello World!')
    '\n    >>> list_times_cint(3)\n    []\n    [None, None, None]\n    [3, 3, 3]\n    [1, 2, 3, 1, 2, 3, 1, 2, 3]\n    '
    py_n = n + 1
    a = [] * py_n
    b = [None] * py_n
    c = py_n * [n]
    d = py_n * [1, 2, 3]
    print(a)
    print(b)
    print(c)
    print(d)

@cython.cfunc
def sideeffect(x) -> cython.int:
    if False:
        while True:
            i = 10
    global _sideeffect_value
    _sideeffect_value += 1
    return _sideeffect_value + x

def reset_sideeffect():
    if False:
        i = 10
        return i + 15
    global _sideeffect_value
    _sideeffect_value = 0

@cython.test_fail_if_path_exists('//MulNode')
@cython.test_assert_path_exists('//ListNode[@mult_factor]')
def complicated_cint_times_list(n: cython.int):
    if False:
        print('Hello World!')
    '\n    >>> complicated_cint_times_list(3)\n    []\n    [None, None, None, None]\n    [3, 3, 3, 3]\n    [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]\n    '
    reset_sideeffect()
    a = [] * sideeffect((lambda : n)())
    reset_sideeffect()
    b = sideeffect((lambda : n)()) * [None]
    reset_sideeffect()
    c = [n] * sideeffect((lambda : n)())
    reset_sideeffect()
    d = sideeffect((lambda : n)()) * [1, 2, 3]
    print(a)
    print(b)
    print(c)
    print(d)