"""
masked_accumulations.py is for accumulation algorithms using a mask-based approach
for missing values.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import numpy as np
if TYPE_CHECKING:
    from pandas._typing import npt

def _cum_func(func: Callable, values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool=True):
    if False:
        return 10
    '\n    Accumulations for 1D masked array.\n\n    We will modify values in place to replace NAs with the appropriate fill value.\n\n    Parameters\n    ----------\n    func : np.cumsum, np.cumprod, np.maximum.accumulate, np.minimum.accumulate\n    values : np.ndarray\n        Numpy array with the values (can be of any dtype that support the\n        operation).\n    mask : np.ndarray\n        Boolean numpy array (True values indicate missing values).\n    skipna : bool, default True\n        Whether to skip NA.\n    '
    dtype_info: np.iinfo | np.finfo
    if values.dtype.kind == 'f':
        dtype_info = np.finfo(values.dtype.type)
    elif values.dtype.kind in 'iu':
        dtype_info = np.iinfo(values.dtype.type)
    elif values.dtype.kind == 'b':
        dtype_info = np.iinfo(np.uint8)
    else:
        raise NotImplementedError(f'No masked accumulation defined for dtype {values.dtype.type}')
    try:
        fill_value = {np.cumprod: 1, np.maximum.accumulate: dtype_info.min, np.cumsum: 0, np.minimum.accumulate: dtype_info.max}[func]
    except KeyError:
        raise NotImplementedError(f'No accumulation for {func} implemented on BaseMaskedArray')
    values[mask] = fill_value
    if not skipna:
        mask = np.maximum.accumulate(mask)
    values = func(values)
    return (values, mask)

def cumsum(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool=True):
    if False:
        return 10
    return _cum_func(np.cumsum, values, mask, skipna=skipna)

def cumprod(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool=True):
    if False:
        i = 10
        return i + 15
    return _cum_func(np.cumprod, values, mask, skipna=skipna)

def cummin(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool=True):
    if False:
        while True:
            i = 10
    return _cum_func(np.minimum.accumulate, values, mask, skipna=skipna)

def cummax(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool=True):
    if False:
        for i in range(10):
            print('nop')
    return _cum_func(np.maximum.accumulate, values, mask, skipna=skipna)