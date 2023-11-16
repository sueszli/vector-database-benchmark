import cmath
import numpy as np
from numba import float32
from numba.types import unicode_type, i8
from numba.pycc import CC, exportmany, export
from numba.tests.support import has_blas
from numba import typed
cc = CC('pycc_test_simple')
cc.use_nrt = False

@cc.export('multf', (float32, float32))
@cc.export('multi', 'i4(i4, i4)')
def mult(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a * b

@cc.export('get_none', 'none()')
def get_none():
    if False:
        return 10
    return None

@cc.export('div', 'f8(f8, f8)')
def div(x, y):
    if False:
        while True:
            i = 10
    return x / y
_two = 2

@cc.export('square', 'i8(i8)')
def square(u):
    if False:
        return 10
    return u ** _two
cc_helperlib = CC('pycc_test_helperlib')
cc_helperlib.use_nrt = False

@cc_helperlib.export('power', 'i8(i8, i8)')
def power(u, v):
    if False:
        i = 10
        return i + 15
    return u ** v

@cc_helperlib.export('sqrt', 'c16(c16)')
def sqrt(u):
    if False:
        i = 10
        return i + 15
    return cmath.sqrt(u)

@cc_helperlib.export('size', 'i8(f8[:])')
def size(arr):
    if False:
        return 10
    return arr.size

@cc_helperlib.export('np_sqrt', 'f8(f8)')
def np_sqrt(u):
    if False:
        while True:
            i = 10
    return np.sqrt(u)

@cc_helperlib.export('spacing', 'f8(f8)')
def np_spacing(u):
    if False:
        for i in range(10):
            print('nop')
    return np.spacing(u)

@cc_helperlib.export('random', 'f8(i4)')
def random_impl(seed):
    if False:
        print('Hello World!')
    if seed != -1:
        np.random.seed(seed)
    return np.random.random()
cc_nrt = CC('pycc_test_nrt')

@cc_nrt.export('zero_scalar', 'f8(i4)')
def zero_scalar(n):
    if False:
        print('Hello World!')
    arr = np.zeros(n)
    return arr[-1]
if has_blas:

    @cc_nrt.export('vector_dot', 'f8(i4)')
    def vector_dot(n):
        if False:
            return 10
        a = np.linspace(1, n, n)
        return np.dot(a, a)

@cc_nrt.export('zeros', 'f8[:](i4)')
def zeros(n):
    if False:
        i = 10
        return i + 15
    return np.zeros(n)

@cc_nrt.export('np_argsort', 'intp[:](float64[:])')
def np_argsort(arr):
    if False:
        for i in range(10):
            print('nop')
    return np.argsort(arr)
exportmany(['multf f4(f4,f4)', 'multi i4(i4,i4)'])(mult)
export('mult f8(f8, f8)')(mult)

@cc_nrt.export('dict_usecase', 'intp[:](intp[:])')
def dict_usecase(arr):
    if False:
        i = 10
        return i + 15
    d = typed.Dict()
    for i in range(arr.size):
        d[i] = arr[i]
    out = np.zeros_like(arr)
    for (k, v) in d.items():
        out[k] = k * v
    return out

@cc_nrt.export('internal_str_dict', i8(unicode_type))
def internal_str_dict(x):
    if False:
        for i in range(10):
            print('nop')
    d = typed.Dict.empty(unicode_type, i8)
    if x not in d:
        d[x] = len(d)
    return len(d)

@cc_nrt.export('hash_str', i8(unicode_type))
def internal_str_dict(x):
    if False:
        for i in range(10):
            print('nop')
    return hash(x)

@cc_nrt.export('hash_literal_str_A', i8())
def internal_str_dict():
    if False:
        print('Hello World!')
    return hash('A')