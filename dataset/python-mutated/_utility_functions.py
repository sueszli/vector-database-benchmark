from __future__ import annotations
from ._array_object import Array
from typing import Optional, Tuple, Union
import cupy as np

def all(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.all <numpy.all>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.asarray(np.all(x._array, axis=axis, keepdims=keepdims)))

def any(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.any <numpy.any>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.asarray(np.any(x._array, axis=axis, keepdims=keepdims)))