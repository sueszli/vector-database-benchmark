import cython

@cython.nogil
@cython.cfunc
def f_nogil(x: cython.int) -> cython.int:
    if False:
        return 10
    y: cython.int
    y = x + 10
    return y

def f_gil(x):
    if False:
        for i in range(10):
            print('nop')
    y = 0
    y = x + 100
    return y

def illegal_gil_usage():
    if False:
        i = 10
        return i + 15
    res: cython.int = 0
    with cython.nogil(True):
        res = f_gil(res)
        with cython.nogil(True):
            res = f_gil(res)
    with cython.nogil(False):
        res = f_nogil(res)

def foo(a):
    if False:
        for i in range(10):
            print('nop')
    return a < 10

def non_constant_condition(x: cython.int) -> cython.int:
    if False:
        print('Hello World!')
    res: cython.int = x
    with cython.nogil(x < 10):
        res = f_nogil(res)
number_or_object = cython.fused_type(cython.float, cython.object)

def fused_type(x: number_or_object):
    if False:
        while True:
            i = 10
    with cython.nogil(number_or_object is object):
        res = x + 1
    with cython.nogil(number_or_object is cython.float):
        res = x + 1
    return res

def nogil_multiple_arguments(x: cython.int) -> cython.int:
    if False:
        for i in range(10):
            print('nop')
    res: cython.int = x
    with cython.nogil(1, 2):
        res = f_nogil(res)

def nogil_keyworkd_arguments(x: cython.int) -> cython.int:
    if False:
        while True:
            i = 10
    res: cython.int = x
    with cython.nogil(kw=2):
        res = f_nogil(res)

@cython.gil(True)
@cython.cfunc
def wrong_decorator() -> cython.int:
    if False:
        i = 10
        return i + 15
    return 0
_ERRORS = u'\n22:14: Accessing Python global or builtin not allowed without gil\n22:19: Calling gil-requiring function not allowed without gil\n22:19: Coercion from Python not allowed without the GIL\n22:19: Constructing Python tuple not allowed without gil\n22:20: Converting to Python object not allowed without gil\n24:13: Trying to release the GIL while it was previously released.\n25:18: Accessing Python global or builtin not allowed without gil\n25:23: Calling gil-requiring function not allowed without gil\n25:23: Coercion from Python not allowed without the GIL\n25:23: Constructing Python tuple not allowed without gil\n25:24: Converting to Python object not allowed without gil\n37:24: Non-constant condition in a `with nogil(<condition>)` statement\n46:8: Assignment of Python object not allowed without gil\n46:16: Calling gil-requiring function not allowed without gil\n56:9: Compiler directive nogil accepts one positional argument.\n61:9: Compiler directive nogil accepts one positional argument.\n65:0: The gil compiler directive is not allowed in function scope\n'