from __future__ import annotations
from ._dtypes import _boolean_dtypes, _floating_dtypes, _integer_dtypes, _integer_or_boolean_dtypes, _numeric_dtypes, _result_type
from ._array_object import Array
import cupy as np

def abs(x: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.abs <numpy.abs>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in abs')
    return Array._new(np.abs(x._array))

def acos(x: Array, /) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.arccos <numpy.arccos>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in acos')
    return Array._new(np.arccos(x._array))

def acosh(x: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.arccosh <numpy.arccosh>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in acosh')
    return Array._new(np.arccosh(x._array))

def add(x1: Array, x2: Array, /) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.add <numpy.add>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in add')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.add(x1._array, x2._array))

def asin(x: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.arcsin <numpy.arcsin>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in asin')
    return Array._new(np.arcsin(x._array))

def asinh(x: Array, /) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.arcsinh <numpy.arcsinh>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in asinh')
    return Array._new(np.arcsinh(x._array))

def atan(x: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.arctan <numpy.arctan>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in atan')
    return Array._new(np.arctan(x._array))

def atan2(x1: Array, x2: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.arctan2 <numpy.arctan2>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in atan2')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.arctan2(x1._array, x2._array))

def atanh(x: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.arctanh <numpy.arctanh>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in atanh')
    return Array._new(np.arctanh(x._array))

def bitwise_and(x1: Array, x2: Array, /) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.bitwise_and <numpy.bitwise_and>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError('Only integer or boolean dtypes are allowed in bitwise_and')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.bitwise_and(x1._array, x2._array))

def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.left_shift <numpy.left_shift>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError('Only integer dtypes are allowed in bitwise_left_shift')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    if np.any(x2._array < 0):
        raise ValueError('bitwise_left_shift(x1, x2) is only defined for x2 >= 0')
    return Array._new(np.left_shift(x1._array, x2._array))

def bitwise_invert(x: Array, /) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.invert <numpy.invert>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _integer_or_boolean_dtypes:
        raise TypeError('Only integer or boolean dtypes are allowed in bitwise_invert')
    return Array._new(np.invert(x._array))

def bitwise_or(x1: Array, x2: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.bitwise_or <numpy.bitwise_or>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError('Only integer or boolean dtypes are allowed in bitwise_or')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.bitwise_or(x1._array, x2._array))

def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.right_shift <numpy.right_shift>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError('Only integer dtypes are allowed in bitwise_right_shift')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    if np.any(x2._array < 0):
        raise ValueError('bitwise_right_shift(x1, x2) is only defined for x2 >= 0')
    return Array._new(np.right_shift(x1._array, x2._array))

def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.bitwise_xor <numpy.bitwise_xor>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError('Only integer or boolean dtypes are allowed in bitwise_xor')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.bitwise_xor(x1._array, x2._array))

def ceil(x: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.ceil <numpy.ceil>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in ceil')
    if x.dtype in _integer_dtypes:
        return x
    return Array._new(np.ceil(x._array))

def cos(x: Array, /) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.cos <numpy.cos>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in cos')
    return Array._new(np.cos(x._array))

def cosh(x: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.cosh <numpy.cosh>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in cosh')
    return Array._new(np.cosh(x._array))

def divide(x1: Array, x2: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.divide <numpy.divide>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in divide')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.divide(x1._array, x2._array))

def equal(x1: Array, x2: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.equal <numpy.equal>`.\n\n    See its docstring for more information.\n    '
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.equal(x1._array, x2._array))

def exp(x: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.exp <numpy.exp>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in exp')
    return Array._new(np.exp(x._array))

def expm1(x: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.expm1 <numpy.expm1>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in expm1')
    return Array._new(np.expm1(x._array))

def floor(x: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.floor <numpy.floor>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in floor')
    if x.dtype in _integer_dtypes:
        return x
    return Array._new(np.floor(x._array))

def floor_divide(x1: Array, x2: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.floor_divide <numpy.floor_divide>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in floor_divide')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.floor_divide(x1._array, x2._array))

def greater(x1: Array, x2: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.greater <numpy.greater>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in greater')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.greater(x1._array, x2._array))

def greater_equal(x1: Array, x2: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.greater_equal <numpy.greater_equal>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in greater_equal')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.greater_equal(x1._array, x2._array))

def isfinite(x: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.isfinite <numpy.isfinite>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in isfinite')
    return Array._new(np.isfinite(x._array))

def isinf(x: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.isinf <numpy.isinf>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in isinf')
    return Array._new(np.isinf(x._array))

def isnan(x: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.isnan <numpy.isnan>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in isnan')
    return Array._new(np.isnan(x._array))

def less(x1: Array, x2: Array, /) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.less <numpy.less>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in less')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.less(x1._array, x2._array))

def less_equal(x1: Array, x2: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.less_equal <numpy.less_equal>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in less_equal')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.less_equal(x1._array, x2._array))

def log(x: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.log <numpy.log>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in log')
    return Array._new(np.log(x._array))

def log1p(x: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.log1p <numpy.log1p>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in log1p')
    return Array._new(np.log1p(x._array))

def log2(x: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.log2 <numpy.log2>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in log2')
    return Array._new(np.log2(x._array))

def log10(x: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.log10 <numpy.log10>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in log10')
    return Array._new(np.log10(x._array))

def logaddexp(x1: Array, x2: Array) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.logaddexp <numpy.logaddexp>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in logaddexp')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.logaddexp(x1._array, x2._array))

def logical_and(x1: Array, x2: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.logical_and <numpy.logical_and>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError('Only boolean dtypes are allowed in logical_and')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.logical_and(x1._array, x2._array))

def logical_not(x: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.logical_not <numpy.logical_not>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _boolean_dtypes:
        raise TypeError('Only boolean dtypes are allowed in logical_not')
    return Array._new(np.logical_not(x._array))

def logical_or(x1: Array, x2: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.logical_or <numpy.logical_or>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError('Only boolean dtypes are allowed in logical_or')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.logical_or(x1._array, x2._array))

def logical_xor(x1: Array, x2: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.logical_xor <numpy.logical_xor>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError('Only boolean dtypes are allowed in logical_xor')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.logical_xor(x1._array, x2._array))

def multiply(x1: Array, x2: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.multiply <numpy.multiply>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in multiply')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.multiply(x1._array, x2._array))

def negative(x: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.negative <numpy.negative>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in negative')
    return Array._new(np.negative(x._array))

def not_equal(x1: Array, x2: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.not_equal <numpy.not_equal>`.\n\n    See its docstring for more information.\n    '
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.not_equal(x1._array, x2._array))

def positive(x: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.positive <numpy.positive>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in positive')
    return Array._new(np.positive(x._array))

def pow(x1: Array, x2: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.power <numpy.power>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in pow')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.power(x1._array, x2._array))

def remainder(x1: Array, x2: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.remainder <numpy.remainder>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in remainder')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.remainder(x1._array, x2._array))

def round(x: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.round <numpy.round>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in round')
    return Array._new(np.round(x._array))

def sign(x: Array, /) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.sign <numpy.sign>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in sign')
    return Array._new(np.sign(x._array))

def sin(x: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.sin <numpy.sin>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in sin')
    return Array._new(np.sin(x._array))

def sinh(x: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.sinh <numpy.sinh>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in sinh')
    return Array._new(np.sinh(x._array))

def square(x: Array, /) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.square <numpy.square>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in square')
    return Array._new(np.square(x._array))

def sqrt(x: Array, /) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.sqrt <numpy.sqrt>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in sqrt')
    return Array._new(np.sqrt(x._array))

def subtract(x1: Array, x2: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.subtract <numpy.subtract>`.\n\n    See its docstring for more information.\n    '
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in subtract')
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.subtract(x1._array, x2._array))

def tan(x: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.tan <numpy.tan>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in tan')
    return Array._new(np.tan(x._array))

def tanh(x: Array, /) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.tanh <numpy.tanh>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in tanh')
    return Array._new(np.tanh(x._array))

def trunc(x: Array, /) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.trunc <numpy.trunc>`.\n\n    See its docstring for more information.\n    '
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in trunc')
    if x.dtype in _integer_dtypes:
        return x
    return Array._new(np.trunc(x._array))