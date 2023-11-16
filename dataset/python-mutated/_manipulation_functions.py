from __future__ import annotations
from ._array_object import Array
from ._data_type_functions import result_type
from typing import List, Optional, Tuple, Union
import cupy as np

def concat(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int]=0) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.concatenate <numpy.concatenate>`.\n\n    See its docstring for more information.\n    '
    dtype = result_type(*arrays)
    arrays = tuple((a._array for a in arrays))
    return Array._new(np.concatenate(arrays, axis=axis, dtype=dtype))

def expand_dims(x: Array, /, *, axis: int) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.expand_dims <numpy.expand_dims>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.expand_dims(x._array, axis))

def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]]=None) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.flip <numpy.flip>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.flip(x._array, axis=axis))

def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.transpose(x._array, axes))

def reshape(x: Array, /, shape: Tuple[int, ...]) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.reshape <numpy.reshape>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.reshape(x._array, shape))

def roll(x: Array, /, shift: Union[int, Tuple[int, ...]], *, axis: Optional[Union[int, Tuple[int, ...]]]=None) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.roll <numpy.roll>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.roll(x._array, shift, axis=axis))

def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.squeeze <numpy.squeeze>`.\n\n    See its docstring for more information.\n    '
    return Array._new(np.squeeze(x._array, axis=axis))

def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int=0) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.stack <numpy.stack>`.\n\n    See its docstring for more information.\n    '
    result_type(*arrays)
    arrays = tuple((a._array for a in arrays))
    return Array._new(np.stack(arrays, axis=axis))