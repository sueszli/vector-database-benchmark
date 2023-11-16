from __future__ import annotations
from ._array_object import Array
import cupy as np

def argsort(x: Array, /, *, axis: int=-1, descending: bool=False, stable: bool=True) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.argsort <numpy.argsort>`.\n\n    See its docstring for more information.\n    '
    kind = None
    if not descending:
        res = np.argsort(x._array, axis=axis, kind=kind)
    else:
        res = np.flip(np.argsort(np.flip(x._array, axis=axis), axis=axis, kind=kind), axis=axis)
        normalised_axis = axis if axis >= 0 else x.ndim + axis
        max_i = x.shape[normalised_axis] - 1
        res = max_i - res
    return Array._new(res)

def sort(x: Array, /, *, axis: int=-1, descending: bool=False, stable: bool=True) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.sort <numpy.sort>`.\n\n    See its docstring for more information.\n    '
    kind = None
    res = np.sort(x._array, axis=axis, kind=kind)
    if descending:
        res = np.flip(res, axis=axis)
    return Array._new(res)