from __future__ import annotations
from ._array_object import Array
from ._dtypes import _all_dtypes, _result_type
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union
if TYPE_CHECKING:
    from ._typing import Dtype
    from collections.abc import Sequence
import cupy as np

def astype(x: Array, dtype: Dtype, /, *, copy: bool=True) -> Array:
    if False:
        print('Hello World!')
    if not copy and dtype == x.dtype:
        return x
    return Array._new(x._array.astype(dtype=dtype, copy=copy))

def broadcast_arrays(*arrays: Array) -> List[Array]:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.broadcast_arrays <numpy.broadcast_arrays>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    return [Array._new(array) for array in np.broadcast_arrays(*[a._array for a in arrays])]

def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.broadcast_to <numpy.broadcast_to>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    return Array._new(np.broadcast_to(x._array, shape))

def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.can_cast <numpy.can_cast>`.\n\n    See its docstring for more information.\n    '
    if isinstance(from_, Array):
        from_ = from_.dtype
    elif from_ not in _all_dtypes:
        raise TypeError(f'from_={from_!r}, but should be an array_api array or dtype')
    if to not in _all_dtypes:
        raise TypeError(f'to={to!r}, but should be a dtype')
    try:
        dtype = _result_type(from_, to)
        return to == dtype
    except TypeError:
        return False

@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float

@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int

def finfo(type: Union[Dtype, Array], /) -> finfo_object:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.finfo <numpy.finfo>`.\n\n    See its docstring for more information.\n    '
    fi = np.finfo(type)
    try:
        tiny = fi.smallest_normal
    except AttributeError:
        tiny = fi.tiny
    return finfo_object(fi.bits, float(fi.eps), float(fi.max), float(fi.min), float(tiny))

def iinfo(type: Union[Dtype, Array], /) -> iinfo_object:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.iinfo <numpy.iinfo>`.\n\n    See its docstring for more information.\n    '
    ii = np.iinfo(type)
    return iinfo_object(ii.bits, ii.max, ii.min)

def result_type(*arrays_and_dtypes: Union[Array, Dtype]) -> Dtype:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.result_type <numpy.result_type>`.\n\n    See its docstring for more information.\n    '
    A = []
    for a in arrays_and_dtypes:
        if isinstance(a, Array):
            a = a.dtype
        elif isinstance(a, np.ndarray) or a not in _all_dtypes:
            raise TypeError('result_type() inputs must be array_api arrays or dtypes')
        A.append(a)
    if len(A) == 0:
        raise ValueError('at least one array or dtype is required')
    elif len(A) == 1:
        return A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
            t = _result_type(t, t2)
        return t