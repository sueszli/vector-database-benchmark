from __future__ import annotations
from ._array_object import Array
from ._dtypes import _result_type
from typing import Optional, Tuple
import cupy as np

def argmax(x: Array, /, *, axis: Optional[int]=None, keepdims: bool=False) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.argmax <numpy.argmax>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.asarray(np.argmax(x._array, axis=axis, keepdims=keepdims)))

def argmin(x: Array, /, *, axis: Optional[int]=None, keepdims: bool=False) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.argmin <numpy.argmin>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.asarray(np.argmin(x._array, axis=axis, keepdims=keepdims)))

def nonzero(x: Array, /) -> Tuple[Array, ...]:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.nonzero <numpy.nonzero>`.\n\n    See its docstring for more information.\n    '
    return tuple((Array._new(i) for i in np.nonzero(x._array)))

def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.where <numpy.where>`.\n\n    See its docstring for more information.\n    '
    _result_type(x1.dtype, x2.dtype)
    (x1, x2) = Array._normalize_two_args(x1, x2)
    return Array._new(np.where(condition._array, x1._array, x2._array))