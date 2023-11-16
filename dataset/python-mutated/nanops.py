from __future__ import annotations
import functools
import itertools
from typing import Any, Callable, cast
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import NaT, NaTType, iNaT, lib
from pandas._typing import ArrayLike, AxisInt, CorrelationMethod, Dtype, DtypeObj, F, Scalar, Shape, npt
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_complex, is_float, is_float_dtype, is_integer, is_numeric_dtype, is_object_dtype, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna
bn = import_optional_dependency('bottleneck', errors='warn')
_BOTTLENECK_INSTALLED = bn is not None
_USE_BOTTLENECK = False

def set_use_bottleneck(v: bool=True) -> None:
    if False:
        i = 10
        return i + 15
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v
set_use_bottleneck(get_option('compute.use_bottleneck'))

class disallow:

    def __init__(self, *dtypes: Dtype) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.dtypes = tuple((pandas_dtype(dtype).type for dtype in dtypes))

    def check(self, obj) -> bool:
        if False:
            i = 10
            return i + 15
        return hasattr(obj, 'dtype') and issubclass(obj.dtype.type, self.dtypes)

    def __call__(self, f: F) -> F:
        if False:
            while True:
                i = 10

        @functools.wraps(f)
        def _f(*args, **kwargs):
            if False:
                print('Hello World!')
            obj_iter = itertools.chain(args, kwargs.values())
            if any((self.check(obj) for obj in obj_iter)):
                f_name = f.__name__.replace('nan', '')
                raise TypeError(f"reduction operation '{f_name}' not allowed for this dtype")
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise
        return cast(F, _f)

class bottleneck_switch:

    def __init__(self, name=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        self.name = name
        self.kwargs = kwargs

    def __call__(self, alt: F) -> F:
        if False:
            print('Hello World!')
        bn_name = self.name or alt.__name__
        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):
            bn_func = None

        @functools.wraps(alt)
        def f(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, **kwds):
            if False:
                while True:
                    i = 10
            if len(self.kwargs) > 0:
                for (k, v) in self.kwargs.items():
                    if k not in kwds:
                        kwds[k] = v
            if values.size == 0 and kwds.get('min_count') is None:
                return _na_for_min_count(values, axis)
            if _USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name):
                if kwds.get('mask', None) is None:
                    kwds.pop('mask', None)
                    result = bn_func(values, axis=axis, **kwds)
                    if _has_infs(result):
                        result = alt(values, axis=axis, skipna=skipna, **kwds)
                else:
                    result = alt(values, axis=axis, skipna=skipna, **kwds)
            else:
                result = alt(values, axis=axis, skipna=skipna, **kwds)
            return result
        return cast(F, f)

def _bn_ok_dtype(dtype: DtypeObj, name: str) -> bool:
    if False:
        i = 10
        return i + 15
    if dtype != object and (not needs_i8_conversion(dtype)):
        return name not in ['nansum', 'nanprod', 'nanmean']
    return False

def _has_infs(result) -> bool:
    if False:
        return 10
    if isinstance(result, np.ndarray):
        if result.dtype in ('f8', 'f4'):
            return lib.has_infs(result.ravel('K'))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        return False

def _get_fill_value(dtype: DtypeObj, fill_value: Scalar | None=None, fill_value_typ=None):
    if False:
        i = 10
        return i + 15
    'return the correct fill value for the dtype of the values'
    if fill_value is not None:
        return fill_value
    if _na_ok_dtype(dtype):
        if fill_value_typ is None:
            return np.nan
        elif fill_value_typ == '+inf':
            return np.inf
        else:
            return -np.inf
    elif fill_value_typ == '+inf':
        return lib.i8max
    else:
        return iNaT

def _maybe_get_mask(values: np.ndarray, skipna: bool, mask: npt.NDArray[np.bool_] | None) -> npt.NDArray[np.bool_] | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute a mask if and only if necessary.\n\n    This function will compute a mask iff it is necessary. Otherwise,\n    return the provided mask (potentially None) when a mask does not need to be\n    computed.\n\n    A mask is never necessary if the values array is of boolean or integer\n    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable\n    dtype that is interpretable as either boolean or integer data (eg,\n    timedelta64), a mask must be provided.\n\n    If the skipna parameter is False, a new mask will not be computed.\n\n    The mask is computed using isna() by default. Setting invert=True selects\n    notna() as the masking function.\n\n    Parameters\n    ----------\n    values : ndarray\n        input array to potentially compute mask for\n    skipna : bool\n        boolean for whether NaNs should be skipped\n    mask : Optional[ndarray]\n        nan-mask if known\n\n    Returns\n    -------\n    Optional[np.ndarray[bool]]\n    '
    if mask is None:
        if values.dtype.kind in 'biu':
            return None
        if skipna or values.dtype.kind in 'mM':
            mask = isna(values)
    return mask

def _get_values(values: np.ndarray, skipna: bool, fill_value: Any=None, fill_value_typ: str | None=None, mask: npt.NDArray[np.bool_] | None=None) -> tuple[np.ndarray, npt.NDArray[np.bool_] | None]:
    if False:
        return 10
    "\n    Utility to get the values view, mask, dtype, dtype_max, and fill_value.\n\n    If both mask and fill_value/fill_value_typ are not None and skipna is True,\n    the values array will be copied.\n\n    For input arrays of boolean or integer dtypes, copies will only occur if a\n    precomputed mask, a fill_value/fill_value_typ, and skipna=True are\n    provided.\n\n    Parameters\n    ----------\n    values : ndarray\n        input array to potentially compute mask for\n    skipna : bool\n        boolean for whether NaNs should be skipped\n    fill_value : Any\n        value to fill NaNs with\n    fill_value_typ : str\n        Set to '+inf' or '-inf' to handle dtype-specific infinities\n    mask : Optional[np.ndarray[bool]]\n        nan-mask if known\n\n    Returns\n    -------\n    values : ndarray\n        Potential copy of input value array\n    mask : Optional[ndarray[bool]]\n        Mask for values, if deemed necessary to compute\n    "
    mask = _maybe_get_mask(values, skipna, mask)
    dtype = values.dtype
    datetimelike = False
    if values.dtype.kind in 'mM':
        values = np.asarray(values.view('i8'))
        datetimelike = True
    if skipna and mask is not None:
        fill_value = _get_fill_value(dtype, fill_value=fill_value, fill_value_typ=fill_value_typ)
        if fill_value is not None:
            if mask.any():
                if datetimelike or _na_ok_dtype(dtype):
                    values = values.copy()
                    np.putmask(values, mask, fill_value)
                else:
                    values = np.where(~mask, values, fill_value)
    return (values, mask)

def _get_dtype_max(dtype: np.dtype) -> np.dtype:
    if False:
        for i in range(10):
            print('nop')
    dtype_max = dtype
    if dtype.kind in 'bi':
        dtype_max = np.dtype(np.int64)
    elif dtype.kind == 'u':
        dtype_max = np.dtype(np.uint64)
    elif dtype.kind == 'f':
        dtype_max = np.dtype(np.float64)
    return dtype_max

def _na_ok_dtype(dtype: DtypeObj) -> bool:
    if False:
        print('Hello World!')
    if needs_i8_conversion(dtype):
        return False
    return not issubclass(dtype.type, np.integer)

def _wrap_results(result, dtype: np.dtype, fill_value=None):
    if False:
        return 10
    'wrap our results if needed'
    if result is NaT:
        pass
    elif dtype.kind == 'M':
        if fill_value is None:
            fill_value = iNaT
        if not isinstance(result, np.ndarray):
            assert not isna(fill_value), 'Expected non-null fill_value'
            if result == fill_value:
                result = np.nan
            if isna(result):
                result = np.datetime64('NaT', 'ns').astype(dtype)
            else:
                result = np.int64(result).view(dtype)
            result = result.astype(dtype, copy=False)
        else:
            result = result.astype(dtype)
    elif dtype.kind == 'm':
        if not isinstance(result, np.ndarray):
            if result == fill_value or np.isnan(result):
                result = np.timedelta64('NaT').astype(dtype)
            elif np.fabs(result) > lib.i8max:
                raise ValueError('overflow in timedelta operation')
            else:
                result = np.int64(result).astype(dtype, copy=False)
        else:
            result = result.astype('m8[ns]').view(dtype)
    return result

def _datetimelike_compat(func: F) -> F:
    if False:
        while True:
            i = 10
    '\n    If we have datetime64 or timedelta64 values, ensure we have a correct\n    mask before calling the wrapped function, then cast back afterwards.\n    '

    @functools.wraps(func)
    def new_func(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None, **kwargs):
        if False:
            while True:
                i = 10
        orig_values = values
        datetimelike = values.dtype.kind in 'mM'
        if datetimelike and mask is None:
            mask = isna(values)
        result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        if datetimelike:
            result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)
            if not skipna:
                assert mask is not None
                result = _mask_datetimelike_result(result, axis, mask, orig_values)
        return result
    return cast(F, new_func)

def _na_for_min_count(values: np.ndarray, axis: AxisInt | None) -> Scalar | np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Return the missing value for `values`.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int or None\n        axis for the reduction, required if values.ndim > 1.\n\n    Returns\n    -------\n    result : scalar or ndarray\n        For 1-D values, returns a scalar of the correct missing type.\n        For 2-D values, returns a 1-D array where each element is missing.\n    '
    if values.dtype.kind in 'iufcb':
        values = values.astype('float64')
    fill_value = na_value_for_dtype(values.dtype)
    if values.ndim == 1:
        return fill_value
    elif axis is None:
        return fill_value
    else:
        result_shape = values.shape[:axis] + values.shape[axis + 1:]
        return np.full(result_shape, fill_value, dtype=values.dtype)

def maybe_operate_rowwise(func: F) -> F:
    if False:
        print('Hello World!')
    '\n    NumPy operations on C-contiguous ndarrays with axis=1 can be\n    very slow if axis 1 >> axis 0.\n    Operate row-by-row and concatenate the results.\n    '

    @functools.wraps(func)
    def newfunc(values: np.ndarray, *, axis: AxisInt | None=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if axis == 1 and values.ndim == 2 and values.flags['C_CONTIGUOUS'] and (values.shape[1] / 1000 > values.shape[0]) and (values.dtype != object) and (values.dtype != bool):
            arrs = list(values)
            if kwargs.get('mask') is not None:
                mask = kwargs.pop('mask')
                results = [func(arrs[i], mask=mask[i], **kwargs) for i in range(len(arrs))]
            else:
                results = [func(x, **kwargs) for x in arrs]
            return np.array(results)
        return func(values, axis=axis, **kwargs)
    return cast(F, newfunc)

def nanany(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Check if any elements along an axis evaluate to True.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : bool\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, 2])\n    >>> nanops.nanany(s.values)\n    True\n\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([np.nan])\n    >>> nanops.nanany(s.values)\n    False\n    '
    if values.dtype.kind in 'iub' and mask is None:
        return values.any(axis)
    if values.dtype.kind == 'M':
        warnings.warn("'any' with datetime64 dtypes is deprecated and will raise in a future version. Use (obj != pd.Timestamp(0)).any() instead.", FutureWarning, stacklevel=find_stack_level())
    (values, _) = _get_values(values, skipna, fill_value=False, mask=mask)
    if values.dtype == object:
        values = values.astype(bool)
    return values.any(axis)

def nanall(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None) -> bool:
    if False:
        while True:
            i = 10
    '\n    Check if all elements along an axis evaluate to True.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : bool\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, 2, np.nan])\n    >>> nanops.nanall(s.values)\n    True\n\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, 0])\n    >>> nanops.nanall(s.values)\n    False\n    '
    if values.dtype.kind in 'iub' and mask is None:
        return values.all(axis)
    if values.dtype.kind == 'M':
        warnings.warn("'all' with datetime64 dtypes is deprecated and will raise in a future version. Use (obj != pd.Timestamp(0)).all() instead.", FutureWarning, stacklevel=find_stack_level())
    (values, _) = _get_values(values, skipna, fill_value=True, mask=mask)
    if values.dtype == object:
        values = values.astype(bool)
    return values.all(axis)

@disallow('M8')
@_datetimelike_compat
@maybe_operate_rowwise
def nansum(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, min_count: int=0, mask: npt.NDArray[np.bool_] | None=None) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Sum the elements along an axis ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray[dtype]\n    axis : int, optional\n    skipna : bool, default True\n    min_count: int, default 0\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : dtype\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, 2, np.nan])\n    >>> nanops.nansum(s.values)\n    3.0\n    '
    dtype = values.dtype
    (values, mask) = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    if dtype.kind == 'f':
        dtype_sum = dtype
    elif dtype.kind == 'm':
        dtype_sum = np.dtype(np.float64)
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)
    return the_sum

def _mask_datetimelike_result(result: np.ndarray | np.datetime64 | np.timedelta64, axis: AxisInt | None, mask: npt.NDArray[np.bool_], orig_values: np.ndarray) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType:
    if False:
        while True:
            i = 10
    if isinstance(result, np.ndarray):
        result = result.astype('i8').view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        result[axis_mask] = iNaT
    elif mask.any():
        return np.int64(iNaT).view(orig_values.dtype)
    return result

@bottleneck_switch()
@_datetimelike_compat
def nanmean(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None) -> float:
    if False:
        return 10
    '\n    Compute the mean of the element along an axis ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    float\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, 2, np.nan])\n    >>> nanops.nanmean(s.values)\n    1.5\n    '
    dtype = values.dtype
    (values, mask) = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    dtype_count = np.dtype(np.float64)
    if dtype.kind in 'mM':
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind in 'iu':
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind == 'f':
        dtype_sum = dtype
        dtype_count = dtype
    count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _ensure_numeric(the_sum)
    if axis is not None and getattr(the_sum, 'ndim', False):
        count = cast(np.ndarray, count)
        with np.errstate(all='ignore'):
            the_mean = the_sum / count
        ct_mask = count == 0
        if ct_mask.any():
            the_mean[ct_mask] = np.nan
    else:
        the_mean = the_sum / count if count > 0 else np.nan
    return the_mean

@bottleneck_switch()
def nanmedian(values, *, axis: AxisInt | None=None, skipna: bool=True, mask=None):
    if False:
        print('Hello World!')
    '\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, np.nan, 2, 2])\n    >>> nanops.nanmedian(s.values)\n    2.0\n    '

    def get_median(x, _mask=None):
        if False:
            while True:
                i = 10
        if _mask is None:
            _mask = notna(x)
        else:
            _mask = ~_mask
        if not skipna and (not _mask.all()):
            return np.nan
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
            res = np.nanmedian(x[_mask])
        return res
    dtype = values.dtype
    (values, mask) = _get_values(values, skipna, mask=mask, fill_value=0)
    if values.dtype.kind != 'f':
        if values.dtype == object:
            inferred = lib.infer_dtype(values)
            if inferred in ['string', 'mixed']:
                raise TypeError(f'Cannot convert {values} to numeric')
        try:
            values = values.astype('f8')
        except ValueError as err:
            raise TypeError(str(err)) from err
    if mask is not None:
        values[mask] = np.nan
    notempty = values.size
    if values.ndim > 1 and axis is not None:
        if notempty:
            if not skipna:
                res = np.apply_along_axis(get_median, axis, values)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
                    if values.shape[1] == 1 and axis == 0 or (values.shape[0] == 1 and axis == 1):
                        res = np.nanmedian(np.squeeze(values), keepdims=True)
                    else:
                        res = np.nanmedian(values, axis=axis)
        else:
            res = _get_empty_reduction_result(values.shape, axis)
    else:
        res = get_median(values, mask) if notempty else np.nan
    return _wrap_results(res, dtype)

def _get_empty_reduction_result(shape: Shape, axis: AxisInt) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    The result from a reduction on an empty ndarray.\n\n    Parameters\n    ----------\n    shape : Tuple[int, ...]\n    axis : int\n\n    Returns\n    -------\n    np.ndarray\n    '
    shp = np.array(shape)
    dims = np.arange(len(shape))
    ret = np.empty(shp[dims != axis], dtype=np.float64)
    ret.fill(np.nan)
    return ret

def _get_counts_nanvar(values_shape: Shape, mask: npt.NDArray[np.bool_] | None, axis: AxisInt | None, ddof: int, dtype: np.dtype=np.dtype(np.float64)) -> tuple[float | np.ndarray, float | np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the count of non-null values along an axis, accounting\n    for degrees of freedom.\n\n    Parameters\n    ----------\n    values_shape : Tuple[int, ...]\n        shape tuple from values ndarray, used if mask is None\n    mask : Optional[ndarray[bool]]\n        locations in values that should be considered missing\n    axis : Optional[int]\n        axis to count along\n    ddof : int\n        degrees of freedom\n    dtype : type, optional\n        type to use for count\n\n    Returns\n    -------\n    count : int, np.nan or np.ndarray\n    d : int, np.nan or np.ndarray\n    '
    count = _get_counts(values_shape, mask, axis, dtype=dtype)
    d = count - dtype.type(ddof)
    if is_float(count):
        if count <= ddof:
            count = np.nan
            d = np.nan
    else:
        count = cast(np.ndarray, count)
        mask = count <= ddof
        if mask.any():
            np.putmask(d, mask, np.nan)
            np.putmask(count, mask, np.nan)
    return (count, d)

@bottleneck_switch(ddof=1)
def nanstd(values, *, axis: AxisInt | None=None, skipna: bool=True, ddof: int=1, mask=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the standard deviation along given axis while ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    ddof : int, default 1\n        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,\n        where N represents the number of elements.\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, np.nan, 2, 3])\n    >>> nanops.nanstd(s.values)\n    1.0\n    '
    if values.dtype == 'M8[ns]':
        values = values.view('m8[ns]')
    orig_dtype = values.dtype
    (values, mask) = _get_values(values, skipna, mask=mask)
    result = np.sqrt(nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask))
    return _wrap_results(result, orig_dtype)

@disallow('M8', 'm8')
@bottleneck_switch(ddof=1)
def nanvar(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, ddof: int=1, mask=None):
    if False:
        print('Hello World!')
    '\n    Compute the variance along given axis while ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    ddof : int, default 1\n        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,\n        where N represents the number of elements.\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, np.nan, 2, 3])\n    >>> nanops.nanvar(s.values)\n    1.0\n    '
    dtype = values.dtype
    mask = _maybe_get_mask(values, skipna, mask)
    if dtype.kind in 'iu':
        values = values.astype('f8')
        if mask is not None:
            values[mask] = np.nan
    if values.dtype.kind == 'f':
        (count, d) = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    else:
        (count, d) = _get_counts_nanvar(values.shape, mask, axis, ddof)
    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    avg = _ensure_numeric(values.sum(axis=axis, dtype=np.float64)) / count
    if axis is not None:
        avg = np.expand_dims(avg, axis)
    sqr = _ensure_numeric((avg - values) ** 2)
    if mask is not None:
        np.putmask(sqr, mask, 0)
    result = sqr.sum(axis=axis, dtype=np.float64) / d
    if dtype.kind == 'f':
        result = result.astype(dtype, copy=False)
    return result

@disallow('M8', 'm8')
def nansem(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, ddof: int=1, mask: npt.NDArray[np.bool_] | None=None) -> float:
    if False:
        i = 10
        return i + 15
    '\n    Compute the standard error in the mean along given axis while ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    ddof : int, default 1\n        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,\n        where N represents the number of elements.\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float64\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, np.nan, 2, 3])\n    >>> nanops.nansem(s.values)\n     0.5773502691896258\n    '
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != 'f':
        values = values.astype('f8')
    if not skipna and mask is not None and mask.any():
        return np.nan
    (count, _) = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)
    return np.sqrt(var) / np.sqrt(count)

def _nanminmax(meth, fill_value_typ):
    if False:
        return 10

    @bottleneck_switch(name=f'nan{meth}')
    @_datetimelike_compat
    def reduction(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None):
        if False:
            while True:
                i = 10
        if values.size == 0:
            return _na_for_min_count(values, axis)
        (values, mask) = _get_values(values, skipna, fill_value_typ=fill_value_typ, mask=mask)
        result = getattr(values, meth)(axis)
        result = _maybe_null_out(result, axis, mask, values.shape)
        return result
    return reduction
nanmin = _nanminmax('min', fill_value_typ='+inf')
nanmax = _nanminmax('max', fill_value_typ='-inf')

def nanargmax(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None) -> int | np.ndarray:
    if False:
        print('Hello World!')
    '\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : int or ndarray[int]\n        The index/indices  of max value in specified axis or -1 in the NA case\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> arr = np.array([1, 2, 3, np.nan, 4])\n    >>> nanops.nanargmax(arr)\n    4\n\n    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)\n    >>> arr[2:, 2] = np.nan\n    >>> arr\n    array([[ 0.,  1.,  2.],\n           [ 3.,  4.,  5.],\n           [ 6.,  7., nan],\n           [ 9., 10., nan]])\n    >>> nanops.nanargmax(arr, axis=1)\n    array([2, 2, 1, 1])\n    '
    (values, mask) = _get_values(values, True, fill_value_typ='-inf', mask=mask)
    result = values.argmax(axis)
    result = _maybe_arg_null_out(result, axis, mask, skipna)
    return result

def nanargmin(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None) -> int | np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : int or ndarray[int]\n        The index/indices of min value in specified axis or -1 in the NA case\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> arr = np.array([1, 2, 3, np.nan, 4])\n    >>> nanops.nanargmin(arr)\n    0\n\n    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)\n    >>> arr[2:, 0] = np.nan\n    >>> arr\n    array([[ 0.,  1.,  2.],\n           [ 3.,  4.,  5.],\n           [nan,  7.,  8.],\n           [nan, 10., 11.]])\n    >>> nanops.nanargmin(arr, axis=1)\n    array([0, 0, 1, 1])\n    '
    (values, mask) = _get_values(values, True, fill_value_typ='+inf', mask=mask)
    result = values.argmin(axis)
    result = _maybe_arg_null_out(result, axis, mask, skipna)
    return result

@disallow('M8', 'm8')
@maybe_operate_rowwise
def nanskew(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the sample skewness.\n\n    The statistic computed here is the adjusted Fisher-Pearson standardized\n    moment coefficient G1. The algorithm computes this coefficient directly\n    from the second and third central moment.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float64\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, np.nan, 1, 2])\n    >>> nanops.nanskew(s.values)\n    1.7320508075688787\n    '
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != 'f':
        values = values.astype('f8')
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)
    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)
    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted ** 2
    adjusted3 = adjusted2 * adjusted
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m3 = adjusted3.sum(axis, dtype=np.float64)
    m2 = _zero_out_fperr(m2)
    m3 = _zero_out_fperr(m3)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = count * (count - 1) ** 0.5 / (count - 2) * (m3 / m2 ** 1.5)
    dtype = values.dtype
    if dtype.kind == 'f':
        result = result.astype(dtype, copy=False)
    if isinstance(result, np.ndarray):
        result = np.where(m2 == 0, 0, result)
        result[count < 3] = np.nan
    else:
        result = dtype.type(0) if m2 == 0 else result
        if count < 3:
            return np.nan
    return result

@disallow('M8', 'm8')
@maybe_operate_rowwise
def nankurt(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None) -> float:
    if False:
        i = 10
        return i + 15
    '\n    Compute the sample excess kurtosis\n\n    The statistic computed here is the adjusted Fisher-Pearson standardized\n    moment coefficient G2, computed directly from the second and fourth\n    central moment.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float64\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, np.nan, 1, 3, 2])\n    >>> nanops.nankurt(s.values)\n    -1.2892561983471076\n    '
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != 'f':
        values = values.astype('f8')
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)
    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)
    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted ** 2
    adjusted4 = adjusted2 ** 2
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m4 = adjusted4.sum(axis, dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = count * (count + 1) * (count - 1) * m4
        denominator = (count - 2) * (count - 3) * m2 ** 2
    numerator = _zero_out_fperr(numerator)
    denominator = _zero_out_fperr(denominator)
    if not isinstance(denominator, np.ndarray):
        if count < 4:
            return np.nan
        if denominator == 0:
            return values.dtype.type(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = numerator / denominator - adj
    dtype = values.dtype
    if dtype.kind == 'f':
        result = result.astype(dtype, copy=False)
    if isinstance(result, np.ndarray):
        result = np.where(denominator == 0, 0, result)
        result[count < 4] = np.nan
    return result

@disallow('M8', 'm8')
@maybe_operate_rowwise
def nanprod(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, min_count: int=0, mask: npt.NDArray[np.bool_] | None=None) -> float:
    if False:
        while True:
            i = 10
    '\n    Parameters\n    ----------\n    values : ndarray[dtype]\n    axis : int, optional\n    skipna : bool, default True\n    min_count: int, default 0\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    Dtype\n        The product of all elements on a given axis. ( NaNs are treated as 1)\n\n    Examples\n    --------\n    >>> from pandas.core import nanops\n    >>> s = pd.Series([1, 2, 3, np.nan])\n    >>> nanops.nanprod(s.values)\n    6.0\n    '
    mask = _maybe_get_mask(values, skipna, mask)
    if skipna and mask is not None:
        values = values.copy()
        values[mask] = 1
    result = values.prod(axis)
    return _maybe_null_out(result, axis, mask, values.shape, min_count=min_count)

def _maybe_arg_null_out(result: np.ndarray, axis: AxisInt | None, mask: npt.NDArray[np.bool_] | None, skipna: bool) -> np.ndarray | int:
    if False:
        for i in range(10):
            print('nop')
    if mask is None:
        return result
    if axis is None or not getattr(result, 'ndim', False):
        if skipna:
            if mask.all():
                return -1
        elif mask.any():
            return -1
    else:
        if skipna:
            na_mask = mask.all(axis)
        else:
            na_mask = mask.any(axis)
        if na_mask.any():
            result[na_mask] = -1
    return result

def _get_counts(values_shape: Shape, mask: npt.NDArray[np.bool_] | None, axis: AxisInt | None, dtype: np.dtype[np.floating]=np.dtype(np.float64)) -> np.floating | npt.NDArray[np.floating]:
    if False:
        while True:
            i = 10
    '\n    Get the count of non-null values along an axis\n\n    Parameters\n    ----------\n    values_shape : tuple of int\n        shape tuple from values ndarray, used if mask is None\n    mask : Optional[ndarray[bool]]\n        locations in values that should be considered missing\n    axis : Optional[int]\n        axis to count along\n    dtype : type, optional\n        type to use for count\n\n    Returns\n    -------\n    count : scalar or array\n    '
    if axis is None:
        if mask is not None:
            n = mask.size - mask.sum()
        else:
            n = np.prod(values_shape)
        return dtype.type(n)
    if mask is not None:
        count = mask.shape[axis] - mask.sum(axis)
    else:
        count = values_shape[axis]
    if is_integer(count):
        return dtype.type(count)
    return count.astype(dtype, copy=False)

def _maybe_null_out(result: np.ndarray | float | NaTType, axis: AxisInt | None, mask: npt.NDArray[np.bool_] | None, shape: tuple[int, ...], min_count: int=1) -> np.ndarray | float | NaTType:
    if False:
        while True:
            i = 10
    '\n    Returns\n    -------\n    Dtype\n        The product of all elements on a given axis. ( NaNs are treated as 1)\n    '
    if mask is None and min_count == 0:
        return result
    if axis is not None and isinstance(result, np.ndarray):
        if mask is not None:
            null_mask = mask.shape[axis] - mask.sum(axis) - min_count < 0
        else:
            below_count = shape[axis] - min_count < 0
            new_shape = shape[:axis] + shape[axis + 1:]
            null_mask = np.broadcast_to(below_count, new_shape)
        if np.any(null_mask):
            if is_numeric_dtype(result):
                if np.iscomplexobj(result):
                    result = result.astype('c16')
                elif not is_float_dtype(result):
                    result = result.astype('f8', copy=False)
                result[null_mask] = np.nan
            else:
                result[null_mask] = None
    elif result is not NaT:
        if check_below_min_count(shape, mask, min_count):
            result_dtype = getattr(result, 'dtype', None)
            if is_float_dtype(result_dtype):
                result = result_dtype.type('nan')
            else:
                result = np.nan
    return result

def check_below_min_count(shape: tuple[int, ...], mask: npt.NDArray[np.bool_] | None, min_count: int) -> bool:
    if False:
        return 10
    '\n    Check for the `min_count` keyword. Returns True if below `min_count` (when\n    missing value should be returned from the reduction).\n\n    Parameters\n    ----------\n    shape : tuple\n        The shape of the values (`values.shape`).\n    mask : ndarray[bool] or None\n        Boolean numpy array (typically of same shape as `shape`) or None.\n    min_count : int\n        Keyword passed through from sum/prod call.\n\n    Returns\n    -------\n    bool\n    '
    if min_count > 0:
        if mask is None:
            non_nulls = np.prod(shape)
        else:
            non_nulls = mask.size - mask.sum()
        if non_nulls < min_count:
            return True
    return False

def _zero_out_fperr(arg):
    if False:
        while True:
            i = 10
    if isinstance(arg, np.ndarray):
        return np.where(np.abs(arg) < 1e-14, 0, arg)
    else:
        return arg.dtype.type(0) if np.abs(arg) < 1e-14 else arg

@disallow('M8', 'm8')
def nancorr(a: np.ndarray, b: np.ndarray, *, method: CorrelationMethod='pearson', min_periods: int | None=None) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    a, b: ndarrays\n    '
    if len(a) != len(b):
        raise AssertionError('Operands to nancorr must have same size')
    if min_periods is None:
        min_periods = 1
    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]
    if len(a) < min_periods:
        return np.nan
    a = _ensure_numeric(a)
    b = _ensure_numeric(b)
    f = get_corr_func(method)
    return f(a, b)

def get_corr_func(method: CorrelationMethod) -> Callable[[np.ndarray, np.ndarray], float]:
    if False:
        while True:
            i = 10
    if method == 'kendall':
        from scipy.stats import kendalltau

        def func(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return kendalltau(a, b)[0]
        return func
    elif method == 'spearman':
        from scipy.stats import spearmanr

        def func(a, b):
            if False:
                i = 10
                return i + 15
            return spearmanr(a, b)[0]
        return func
    elif method == 'pearson':

        def func(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return np.corrcoef(a, b)[0, 1]
        return func
    elif callable(method):
        return method
    raise ValueError(f"Unknown method '{method}', expected one of 'kendall', 'spearman', 'pearson', or callable")

@disallow('M8', 'm8')
def nancov(a: np.ndarray, b: np.ndarray, *, min_periods: int | None=None, ddof: int | None=1) -> float:
    if False:
        for i in range(10):
            print('nop')
    if len(a) != len(b):
        raise AssertionError('Operands to nancov must have same size')
    if min_periods is None:
        min_periods = 1
    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]
    if len(a) < min_periods:
        return np.nan
    a = _ensure_numeric(a)
    b = _ensure_numeric(b)
    return np.cov(a, b, ddof=ddof)[0, 1]

def _ensure_numeric(x):
    if False:
        while True:
            i = 10
    if isinstance(x, np.ndarray):
        if x.dtype.kind in 'biu':
            x = x.astype(np.float64)
        elif x.dtype == object:
            inferred = lib.infer_dtype(x)
            if inferred in ['string', 'mixed']:
                raise TypeError(f'Could not convert {x} to numeric')
            try:
                x = x.astype(np.complex128)
            except (TypeError, ValueError):
                try:
                    x = x.astype(np.float64)
                except ValueError as err:
                    raise TypeError(f'Could not convert {x} to numeric') from err
            else:
                if not np.any(np.imag(x)):
                    x = x.real
    elif not (is_float(x) or is_integer(x) or is_complex(x)):
        if isinstance(x, str):
            raise TypeError(f"Could not convert string '{x}' to numeric")
        try:
            x = float(x)
        except (TypeError, ValueError):
            try:
                x = complex(x)
            except ValueError as err:
                raise TypeError(f'Could not convert {x} to numeric') from err
    return x

def na_accum_func(values: ArrayLike, accum_func, *, skipna: bool) -> ArrayLike:
    if False:
        while True:
            i = 10
    '\n    Cumulative function with skipna support.\n\n    Parameters\n    ----------\n    values : np.ndarray or ExtensionArray\n    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}\n    skipna : bool\n\n    Returns\n    -------\n    np.ndarray or ExtensionArray\n    '
    (mask_a, mask_b) = {np.cumprod: (1.0, np.nan), np.maximum.accumulate: (-np.inf, np.nan), np.cumsum: (0.0, np.nan), np.minimum.accumulate: (np.inf, np.nan)}[accum_func]
    assert values.dtype.kind not in 'mM'
    if skipna and (not issubclass(values.dtype.type, (np.integer, np.bool_))):
        vals = values.copy()
        mask = isna(vals)
        vals[mask] = mask_a
        result = accum_func(vals, axis=0)
        result[mask] = mask_b
    else:
        result = accum_func(values, axis=0)
    return result