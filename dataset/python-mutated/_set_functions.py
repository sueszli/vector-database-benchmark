from __future__ import annotations
from ._array_object import Array
from typing import NamedTuple
import cupy as np

class UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array

class UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array

class UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array

def unique_all(x: Array, /) -> UniqueAllResult:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.\n\n    See its docstring for more information.\n    '
    (values, indices, inverse_indices, counts) = np.unique(x._array, return_counts=True, return_index=True, return_inverse=True, equal_nan=False)
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueAllResult(Array._new(values), Array._new(indices), Array._new(inverse_indices), Array._new(counts))

def unique_counts(x: Array, /) -> UniqueCountsResult:
    if False:
        print('Hello World!')
    res = np.unique(x._array, return_counts=True, return_index=False, return_inverse=False, equal_nan=False)
    return UniqueCountsResult(*[Array._new(i) for i in res])

def unique_inverse(x: Array, /) -> UniqueInverseResult:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.\n\n    See its docstring for more information.\n    '
    (values, inverse_indices) = np.unique(x._array, return_counts=False, return_index=False, return_inverse=True, equal_nan=False)
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueInverseResult(Array._new(values), Array._new(inverse_indices))

def unique_values(x: Array, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.\n\n    See its docstring for more information.\n    '
    res = np.unique(x._array, return_counts=False, return_index=False, return_inverse=False, equal_nan=False)
    return Array._new(res)