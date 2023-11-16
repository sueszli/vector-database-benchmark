from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from numpy.core.multiarray import normalize_axis_index
from packaging.version import Version
try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning
from xarray.core.options import OPTIONS
from xarray.core.pycompat import is_duck_array
try:
    import bottleneck as bn
    _BOTTLENECK_AVAILABLE = True
except ImportError:
    bn = np
    _BOTTLENECK_AVAILABLE = False
try:
    import numbagg
    _HAS_NUMBAGG = Version(numbagg.__version__) >= Version('0.5.0')
except ImportError:
    numbagg = np
    _HAS_NUMBAGG = False

def _select_along_axis(values, idx, axis):
    if False:
        for i in range(10):
            print('nop')
    other_ind = np.ix_(*[np.arange(s) for s in idx.shape])
    sl = other_ind[:axis] + (idx,) + other_ind[axis:]
    return values[sl]

def nanfirst(values, axis, keepdims=False):
    if False:
        i = 10
        return i + 15
    if isinstance(axis, tuple):
        (axis,) = axis
    axis = normalize_axis_index(axis, values.ndim)
    idx_first = np.argmax(~pd.isnull(values), axis=axis)
    result = _select_along_axis(values, idx_first, axis)
    if keepdims:
        return np.expand_dims(result, axis=axis)
    else:
        return result

def nanlast(values, axis, keepdims=False):
    if False:
        while True:
            i = 10
    if isinstance(axis, tuple):
        (axis,) = axis
    axis = normalize_axis_index(axis, values.ndim)
    rev = (slice(None),) * axis + (slice(None, None, -1),)
    idx_last = -1 - np.argmax(~pd.isnull(values)[rev], axis=axis)
    result = _select_along_axis(values, idx_last, axis)
    if keepdims:
        return np.expand_dims(result, axis=axis)
    else:
        return result

def inverse_permutation(indices: np.ndarray, N: int | None=None) -> np.ndarray:
    if False:
        print('Hello World!')
    'Return indices for an inverse permutation.\n\n    Parameters\n    ----------\n    indices : 1D np.ndarray with dtype=int\n        Integer positions to assign elements to.\n    N : int, optional\n        Size of the array\n\n    Returns\n    -------\n    inverse_permutation : 1D np.ndarray with dtype=int\n        Integer indices to take from the original array to create the\n        permutation.\n    '
    if N is None:
        N = len(indices)
    inverse_permutation = np.full(N, -1, dtype=np.intp)
    inverse_permutation[indices] = np.arange(len(indices), dtype=np.intp)
    return inverse_permutation

def _ensure_bool_is_ndarray(result, *args):
    if False:
        i = 10
        return i + 15
    if isinstance(result, bool):
        shape = np.broadcast(*args).shape
        constructor = np.ones if result else np.zeros
        result = constructor(shape, dtype=bool)
    return result

def array_eq(self, other):
    if False:
        while True:
            i = 10
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'elementwise comparison failed')
        return _ensure_bool_is_ndarray(self == other, self, other)

def array_ne(self, other):
    if False:
        return 10
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'elementwise comparison failed')
        return _ensure_bool_is_ndarray(self != other, self, other)

def _is_contiguous(positions):
    if False:
        print('Hello World!')
    'Given a non-empty list, does it consist of contiguous integers?'
    previous = positions[0]
    for current in positions[1:]:
        if current != previous + 1:
            return False
        previous = current
    return True

def _advanced_indexer_subspaces(key):
    if False:
        while True:
            i = 10
    'Indices of the advanced indexes subspaces for mixed indexing and vindex.'
    if not isinstance(key, tuple):
        key = (key,)
    advanced_index_positions = [i for (i, k) in enumerate(key) if not isinstance(k, slice)]
    if not advanced_index_positions or not _is_contiguous(advanced_index_positions):
        return ((), ())
    non_slices = [k for k in key if not isinstance(k, slice)]
    broadcasted_shape = np.broadcast_shapes(*[item.shape if is_duck_array(item) else (0,) for item in non_slices])
    ndim = len(broadcasted_shape)
    mixed_positions = advanced_index_positions[0] + np.arange(ndim)
    vindex_positions = np.arange(ndim)
    return (mixed_positions, vindex_positions)

class NumpyVIndexAdapter:
    """Object that implements indexing like vindex on a np.ndarray.

    This is a pure Python implementation of (some of) the logic in this NumPy
    proposal: https://github.com/numpy/numpy/pull/6256
    """

    def __init__(self, array):
        if False:
            for i in range(10):
                print('nop')
        self._array = array

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        (mixed_positions, vindex_positions) = _advanced_indexer_subspaces(key)
        return np.moveaxis(self._array[key], mixed_positions, vindex_positions)

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        'Value must have dimensionality matching the key.'
        (mixed_positions, vindex_positions) = _advanced_indexer_subspaces(key)
        self._array[key] = np.moveaxis(value, vindex_positions, mixed_positions)

def _create_method(name, npmodule=np):
    if False:
        return 10

    def f(values, axis=None, **kwargs):
        if False:
            i = 10
            return i + 15
        dtype = kwargs.get('dtype', None)
        bn_func = getattr(bn, name, None)
        nba_func = getattr(numbagg, name, None)
        if _HAS_NUMBAGG and OPTIONS['use_numbagg'] and isinstance(values, np.ndarray) and (nba_func is not None) and (('var' in name or 'std' in name) and kwargs.get('ddof', 0) == 1) and (values.dtype.kind in 'uifc') and (dtype is None or np.dtype(dtype) == values.dtype):
            kwargs.pop('dtype', None)
            kwargs.pop('ddof', None)
            result = nba_func(values, axis=axis, **kwargs)
        elif _BOTTLENECK_AVAILABLE and OPTIONS['use_bottleneck'] and isinstance(values, np.ndarray) and (bn_func is not None) and (not isinstance(axis, tuple)) and (values.dtype.kind in 'uifc') and values.dtype.isnative and (dtype is None or np.dtype(dtype) == values.dtype):
            kwargs.pop('dtype', None)
            result = bn_func(values, axis=axis, **kwargs)
        else:
            result = getattr(npmodule, name)(values, axis=axis, **kwargs)
        return result
    f.__name__ = name
    return f

def _nanpolyfit_1d(arr, x, rcond=None):
    if False:
        print('Hello World!')
    out = np.full((x.shape[1] + 1,), np.nan)
    mask = np.isnan(arr)
    if not np.all(mask):
        (out[:-1], resid, rank, _) = np.linalg.lstsq(x[~mask, :], arr[~mask], rcond=rcond)
        out[-1] = resid if resid.size > 0 else np.nan
        warn_on_deficient_rank(rank, x.shape[1])
    return out

def warn_on_deficient_rank(rank, order):
    if False:
        print('Hello World!')
    if rank != order:
        warnings.warn('Polyfit may be poorly conditioned', RankWarning, stacklevel=2)

def least_squares(lhs, rhs, rcond=None, skipna=False):
    if False:
        while True:
            i = 10
    if skipna:
        added_dim = rhs.ndim == 1
        if added_dim:
            rhs = rhs.reshape(rhs.shape[0], 1)
        nan_cols = np.any(np.isnan(rhs), axis=0)
        out = np.empty((lhs.shape[1] + 1, rhs.shape[1]))
        if np.any(nan_cols):
            out[:, nan_cols] = np.apply_along_axis(_nanpolyfit_1d, 0, rhs[:, nan_cols], lhs)
        if np.any(~nan_cols):
            (out[:-1, ~nan_cols], resids, rank, _) = np.linalg.lstsq(lhs, rhs[:, ~nan_cols], rcond=rcond)
            out[-1, ~nan_cols] = resids if resids.size > 0 else np.nan
            warn_on_deficient_rank(rank, lhs.shape[1])
        coeffs = out[:-1, :]
        residuals = out[-1, :]
        if added_dim:
            coeffs = coeffs.reshape(coeffs.shape[0])
            residuals = residuals.reshape(residuals.shape[0])
    else:
        (coeffs, residuals, rank, _) = np.linalg.lstsq(lhs, rhs, rcond=rcond)
        if residuals.size == 0:
            residuals = coeffs[0] * np.nan
        warn_on_deficient_rank(rank, lhs.shape[1])
    return (coeffs, residuals)
nanmin = _create_method('nanmin')
nanmax = _create_method('nanmax')
nanmean = _create_method('nanmean')
nanmedian = _create_method('nanmedian')
nanvar = _create_method('nanvar')
nanstd = _create_method('nanstd')
nanprod = _create_method('nanprod')
nancumsum = _create_method('nancumsum')
nancumprod = _create_method('nancumprod')
nanargmin = _create_method('nanargmin')
nanargmax = _create_method('nanargmax')