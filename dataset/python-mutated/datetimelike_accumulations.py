"""
datetimelke_accumulations.py is for accumulations of datetimelike extension arrays
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from pandas._libs import iNaT
from pandas.core.dtypes.missing import isna

def _cum_func(func: Callable, values: np.ndarray, *, skipna: bool=True):
    if False:
        while True:
            i = 10
    '\n    Accumulations for 1D datetimelike arrays.\n\n    Parameters\n    ----------\n    func : np.cumsum, np.maximum.accumulate, np.minimum.accumulate\n    values : np.ndarray\n        Numpy array with the values (can be of any dtype that support the\n        operation). Values is changed is modified inplace.\n    skipna : bool, default True\n        Whether to skip NA.\n    '
    try:
        fill_value = {np.maximum.accumulate: np.iinfo(np.int64).min, np.cumsum: 0, np.minimum.accumulate: np.iinfo(np.int64).max}[func]
    except KeyError:
        raise ValueError(f'No accumulation for {func} implemented on BaseMaskedArray')
    mask = isna(values)
    y = values.view('i8')
    y[mask] = fill_value
    if not skipna:
        mask = np.maximum.accumulate(mask)
    result = func(y)
    result[mask] = iNaT
    if values.dtype.kind in 'mM':
        return result.view(values.dtype.base)
    return result

def cumsum(values: np.ndarray, *, skipna: bool=True) -> np.ndarray:
    if False:
        while True:
            i = 10
    return _cum_func(np.cumsum, values, skipna=skipna)

def cummin(values: np.ndarray, *, skipna: bool=True):
    if False:
        print('Hello World!')
    return _cum_func(np.minimum.accumulate, values, skipna=skipna)

def cummax(values: np.ndarray, *, skipna: bool=True):
    if False:
        for i in range(10):
            print('nop')
    return _cum_func(np.maximum.accumulate, values, skipna=skipna)