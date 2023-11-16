"""Compatibility module defining operations on duck numpy-arrays.

Currently, this means Dask or NumPy arrays. None of these functions should
accept or return xarray objects.
"""
from __future__ import annotations
import contextlib
import datetime
import inspect
import warnings
from functools import partial
from importlib import import_module
import numpy as np
import pandas as pd
from numpy import all as array_all
from numpy import any as array_any
from numpy import around, gradient, isclose, isin, isnat, take, tensordot, transpose, unravel_index, zeros_like
from numpy import concatenate as _concatenate
from numpy.core.multiarray import normalize_axis_index
from numpy.lib.stride_tricks import sliding_window_view
from xarray.core import dask_array_ops, dtypes, nputils
from xarray.core.parallelcompat import get_chunked_array_type, is_chunked_array
from xarray.core.pycompat import array_type, is_duck_dask_array
from xarray.core.utils import is_duck_array, module_available
dask_available = module_available('dask')

def get_array_namespace(x):
    if False:
        while True:
            i = 10
    if hasattr(x, '__array_namespace__'):
        return x.__array_namespace__()
    else:
        return np

def einsum(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    from xarray.core.options import OPTIONS
    if OPTIONS['use_opt_einsum'] and module_available('opt_einsum'):
        import opt_einsum
        return opt_einsum.contract(*args, **kwargs)
    else:
        return np.einsum(*args, **kwargs)

def _dask_or_eager_func(name, eager_module=np, dask_module='dask.array'):
    if False:
        for i in range(10):
            print('nop')
    'Create a function that dispatches to dask for dask array inputs.'

    def f(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        if any((is_duck_dask_array(a) for a in args)):
            mod = import_module(dask_module) if isinstance(dask_module, str) else dask_module
            wrapped = getattr(mod, name)
        else:
            wrapped = getattr(eager_module, name)
        return wrapped(*args, **kwargs)
    return f

def fail_on_dask_array_input(values, msg=None, func_name=None):
    if False:
        return 10
    if is_duck_dask_array(values):
        if msg is None:
            msg = '%r is not yet a valid method on dask arrays'
        if func_name is None:
            func_name = inspect.stack()[1][3]
        raise NotImplementedError(msg % func_name)
pandas_isnull = _dask_or_eager_func('isnull', eager_module=pd, dask_module='dask.array')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.,  2.])', 'array([0., 2.])')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.,  2.])', 'array([0., 2.])')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.4,  1.6])', 'array([0.4, 1.6])')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.,  2.,  2.,  4.,  4.])', 'array([0., 2., 2., 4., 4.])')
around.__doc__ = str.replace(around.__doc__ or '', '    .. [2] "How Futile are Mindless Assessments of\n           Roundoff in Floating-Point Computation?", William Kahan,\n           https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf\n', '')

def isnull(data):
    if False:
        while True:
            i = 10
    data = asarray(data)
    scalar_type = data.dtype.type
    if issubclass(scalar_type, (np.datetime64, np.timedelta64)):
        return isnat(data)
    elif issubclass(scalar_type, np.inexact):
        xp = get_array_namespace(data)
        return xp.isnan(data)
    elif issubclass(scalar_type, (np.bool_, np.integer, np.character, np.void)):
        return zeros_like(data, dtype=bool)
    elif isinstance(data, np.ndarray):
        return pandas_isnull(data)
    else:
        return data != data

def notnull(data):
    if False:
        i = 10
        return i + 15
    return ~isnull(data)
masked_invalid = _dask_or_eager_func('masked_invalid', eager_module=np.ma, dask_module='dask.array.ma')

def trapz(y, x, axis):
    if False:
        i = 10
        return i + 15
    if axis < 0:
        axis = y.ndim + axis
    x_sl1 = (slice(1, None),) + (None,) * (y.ndim - axis - 1)
    x_sl2 = (slice(None, -1),) + (None,) * (y.ndim - axis - 1)
    slice1 = (slice(None),) * axis + (slice(1, None),)
    slice2 = (slice(None),) * axis + (slice(None, -1),)
    dx = x[x_sl1] - x[x_sl2]
    integrand = dx * 0.5 * (y[tuple(slice1)] + y[tuple(slice2)])
    return sum(integrand, axis=axis, skipna=False)

def cumulative_trapezoid(y, x, axis):
    if False:
        i = 10
        return i + 15
    if axis < 0:
        axis = y.ndim + axis
    x_sl1 = (slice(1, None),) + (None,) * (y.ndim - axis - 1)
    x_sl2 = (slice(None, -1),) + (None,) * (y.ndim - axis - 1)
    slice1 = (slice(None),) * axis + (slice(1, None),)
    slice2 = (slice(None),) * axis + (slice(None, -1),)
    dx = x[x_sl1] - x[x_sl2]
    integrand = dx * 0.5 * (y[tuple(slice1)] + y[tuple(slice2)])
    pads = [(1, 0) if i == axis else (0, 0) for i in range(y.ndim)]
    integrand = np.pad(integrand, pads, mode='constant', constant_values=0.0)
    return cumsum(integrand, axis=axis, skipna=False)

def astype(data, dtype, **kwargs):
    if False:
        return 10
    if hasattr(data, '__array_namespace__'):
        xp = get_array_namespace(data)
        if xp == np:
            return data.astype(dtype, **kwargs)
        return xp.astype(data, dtype, **kwargs)
    return data.astype(dtype, **kwargs)

def asarray(data, xp=np):
    if False:
        for i in range(10):
            print('nop')
    return data if is_duck_array(data) else xp.asarray(data)

def as_shared_dtype(scalars_or_arrays, xp=np):
    if False:
        i = 10
        return i + 15
    "Cast a arrays to a shared dtype using xarray's type promotion rules."
    array_type_cupy = array_type('cupy')
    if array_type_cupy and any((isinstance(x, array_type_cupy) for x in scalars_or_arrays)):
        import cupy as cp
        arrays = [asarray(x, xp=cp) for x in scalars_or_arrays]
    else:
        arrays = [asarray(x, xp=xp) for x in scalars_or_arrays]
    out_type = dtypes.result_type(*arrays)
    return [astype(x, out_type, copy=False) for x in arrays]

def broadcast_to(array, shape):
    if False:
        i = 10
        return i + 15
    xp = get_array_namespace(array)
    return xp.broadcast_to(array, shape)

def lazy_array_equiv(arr1, arr2):
    if False:
        for i in range(10):
            print('nop')
    "Like array_equal, but doesn't actually compare values.\n    Returns True when arr1, arr2 identical or their dask tokens are equal.\n    Returns False when shapes are not equal.\n    Returns None when equality cannot determined: one or both of arr1, arr2 are numpy arrays;\n    or their dask tokens are not equal\n    "
    if arr1 is arr2:
        return True
    arr1 = asarray(arr1)
    arr2 = asarray(arr2)
    if arr1.shape != arr2.shape:
        return False
    if dask_available and is_duck_dask_array(arr1) and is_duck_dask_array(arr2):
        from dask.base import tokenize
        if tokenize(arr1) == tokenize(arr2):
            return True
        else:
            return None
    return None

def allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08):
    if False:
        print('Hello World!')
    'Like np.allclose, but also allows values to be NaN in both arrays'
    arr1 = asarray(arr1)
    arr2 = asarray(arr2)
    lazy_equiv = lazy_array_equiv(arr1, arr2)
    if lazy_equiv is None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
            return bool(isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True).all())
    else:
        return lazy_equiv

def array_equiv(arr1, arr2):
    if False:
        for i in range(10):
            print('nop')
    'Like np.array_equal, but also allows values to be NaN in both arrays'
    arr1 = asarray(arr1)
    arr2 = asarray(arr2)
    lazy_equiv = lazy_array_equiv(arr1, arr2)
    if lazy_equiv is None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', "In the future, 'NAT == x'")
            flag_array = (arr1 == arr2) | isnull(arr1) & isnull(arr2)
            return bool(flag_array.all())
    else:
        return lazy_equiv

def array_notnull_equiv(arr1, arr2):
    if False:
        while True:
            i = 10
    'Like np.array_equal, but also allows values to be NaN in either or both\n    arrays\n    '
    arr1 = asarray(arr1)
    arr2 = asarray(arr2)
    lazy_equiv = lazy_array_equiv(arr1, arr2)
    if lazy_equiv is None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', "In the future, 'NAT == x'")
            flag_array = (arr1 == arr2) | isnull(arr1) | isnull(arr2)
            return bool(flag_array.all())
    else:
        return lazy_equiv

def count(data, axis=None):
    if False:
        return 10
    'Count the number of non-NA in this array along the given axis or axes'
    return np.sum(np.logical_not(isnull(data)), axis=axis)

def sum_where(data, axis=None, dtype=None, where=None):
    if False:
        print('Hello World!')
    xp = get_array_namespace(data)
    if where is not None:
        a = where_method(xp.zeros_like(data), where, data)
    else:
        a = data
    result = xp.sum(a, axis=axis, dtype=dtype)
    return result

def where(condition, x, y):
    if False:
        i = 10
        return i + 15
    'Three argument where() with better dtype promotion rules.'
    xp = get_array_namespace(condition)
    return xp.where(condition, *as_shared_dtype([x, y], xp=xp))

def where_method(data, cond, other=dtypes.NA):
    if False:
        for i in range(10):
            print('nop')
    if other is dtypes.NA:
        other = dtypes.get_fill_value(data.dtype)
    return where(cond, data, other)

def fillna(data, other):
    if False:
        return 10
    return where(notnull(data), data, other)

def concatenate(arrays, axis=0):
    if False:
        print('Hello World!')
    'concatenate() with better dtype promotion rules.'
    if hasattr(arrays[0], '__array_namespace__'):
        xp = get_array_namespace(arrays[0])
        return xp.concat(as_shared_dtype(arrays, xp=xp), axis=axis)
    return _concatenate(as_shared_dtype(arrays), axis=axis)

def stack(arrays, axis=0):
    if False:
        for i in range(10):
            print('nop')
    'stack() with better dtype promotion rules.'
    xp = get_array_namespace(arrays[0])
    return xp.stack(as_shared_dtype(arrays, xp=xp), axis=axis)

def reshape(array, shape):
    if False:
        return 10
    xp = get_array_namespace(array)
    return xp.reshape(array, shape)

def ravel(array):
    if False:
        for i in range(10):
            print('nop')
    return reshape(array, (-1,))

@contextlib.contextmanager
def _ignore_warnings_if(condition):
    if False:
        print('Hello World!')
    if condition:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    else:
        yield

def _create_nan_agg_method(name, coerce_strings=False, invariant_0d=False):
    if False:
        while True:
            i = 10
    from xarray.core import nanops

    def f(values, axis=None, skipna=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if kwargs.pop('out', None) is not None:
            raise TypeError(f'`out` is not valid for {name}')
        if invariant_0d and axis == ():
            return values
        values = asarray(values)
        if coerce_strings and values.dtype.kind in 'SU':
            values = astype(values, object)
        func = None
        if skipna or (skipna is None and values.dtype.kind in 'cfO'):
            nanname = 'nan' + name
            func = getattr(nanops, nanname)
        else:
            if name in ['sum', 'prod']:
                kwargs.pop('min_count', None)
            xp = get_array_namespace(values)
            func = getattr(xp, name)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'All-NaN slice encountered')
                return func(values, axis=axis, **kwargs)
        except AttributeError:
            if not is_duck_dask_array(values):
                raise
            try:
                return func(values, axis=axis, dtype=values.dtype, **kwargs)
            except (AttributeError, TypeError):
                raise NotImplementedError(f'{name} is not yet implemented on dask arrays')
    f.__name__ = name
    return f
argmax = _create_nan_agg_method('argmax', coerce_strings=True)
argmin = _create_nan_agg_method('argmin', coerce_strings=True)
max = _create_nan_agg_method('max', coerce_strings=True, invariant_0d=True)
min = _create_nan_agg_method('min', coerce_strings=True, invariant_0d=True)
sum = _create_nan_agg_method('sum', invariant_0d=True)
sum.numeric_only = True
sum.available_min_count = True
std = _create_nan_agg_method('std')
std.numeric_only = True
var = _create_nan_agg_method('var')
var.numeric_only = True
median = _create_nan_agg_method('median', invariant_0d=True)
median.numeric_only = True
prod = _create_nan_agg_method('prod', invariant_0d=True)
prod.numeric_only = True
prod.available_min_count = True
cumprod_1d = _create_nan_agg_method('cumprod', invariant_0d=True)
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method('cumsum', invariant_0d=True)
cumsum_1d.numeric_only = True
_mean = _create_nan_agg_method('mean', invariant_0d=True)

def _datetime_nanmin(array):
    if False:
        i = 10
        return i + 15
    "nanmin() function for datetime64.\n\n    Caveats that this function deals with:\n\n    - In numpy < 1.18, min() on datetime64 incorrectly ignores NaT\n    - numpy nanmin() don't work on datetime64 (all versions at the moment of writing)\n    - dask min() does not work on datetime64 (all versions at the moment of writing)\n    "
    assert array.dtype.kind in 'mM'
    dtype = array.dtype
    array = where(pandas_isnull(array), np.nan, array.astype(float))
    array = min(array, skipna=True)
    if isinstance(array, float):
        array = np.array(array)
    return array.astype(dtype)

def datetime_to_numeric(array, offset=None, datetime_unit=None, dtype=float):
    if False:
        i = 10
        return i + 15
    "Convert an array containing datetime-like data to numerical values.\n    Convert the datetime array to a timedelta relative to an offset.\n    Parameters\n    ----------\n    array : array-like\n        Input data\n    offset : None, datetime or cftime.datetime\n        Datetime offset. If None, this is set by default to the array's minimum\n        value to reduce round off errors.\n    datetime_unit : {None, Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}\n        If not None, convert output to a given datetime unit. Note that some\n        conversions are not allowed due to non-linear relationships between units.\n    dtype : dtype\n        Output dtype.\n    Returns\n    -------\n    array\n        Numerical representation of datetime object relative to an offset.\n    Notes\n    -----\n    Some datetime unit conversions won't work, for example from days to years, even\n    though some calendars would allow for them (e.g. no_leap). This is because there\n    is no `cftime.timedelta` object.\n    "
    if offset is None:
        if array.dtype.kind in 'Mm':
            offset = _datetime_nanmin(array)
        else:
            offset = min(array)
    if is_duck_dask_array(array) and np.issubdtype(array.dtype, object):
        array = array.map_blocks(lambda a, b: a - b, offset, meta=array._meta)
    else:
        array = array - offset
    if not hasattr(array, 'dtype'):
        array = np.array(array)
    if array.dtype.kind in 'O':
        return py_timedelta_to_float(array, datetime_unit or 'ns').astype(dtype)
    elif array.dtype.kind in 'mM':
        if datetime_unit:
            array = array / np.timedelta64(1, datetime_unit)
        return np.where(isnull(array), np.nan, array.astype(dtype))

def timedelta_to_numeric(value, datetime_unit='ns', dtype=float):
    if False:
        i = 10
        return i + 15
    'Convert a timedelta-like object to numerical values.\n\n    Parameters\n    ----------\n    value : datetime.timedelta, numpy.timedelta64, pandas.Timedelta, str\n        Time delta representation.\n    datetime_unit : {Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}\n        The time units of the output values. Note that some conversions are not allowed due to\n        non-linear relationships between units.\n    dtype : type\n        The output data type.\n\n    '
    import datetime as dt
    if isinstance(value, dt.timedelta):
        out = py_timedelta_to_float(value, datetime_unit)
    elif isinstance(value, np.timedelta64):
        out = np_timedelta64_to_float(value, datetime_unit)
    elif isinstance(value, pd.Timedelta):
        out = pd_timedelta_to_float(value, datetime_unit)
    elif isinstance(value, str):
        try:
            a = pd.to_timedelta(value)
        except ValueError:
            raise ValueError(f'Could not convert {value!r} to timedelta64 using pandas.to_timedelta')
        return py_timedelta_to_float(a, datetime_unit)
    else:
        raise TypeError(f'Expected value of type str, pandas.Timedelta, datetime.timedelta or numpy.timedelta64, but received {type(value).__name__}')
    return out.astype(dtype)

def _to_pytimedelta(array, unit='us'):
    if False:
        i = 10
        return i + 15
    return array.astype(f'timedelta64[{unit}]').astype(datetime.timedelta)

def np_timedelta64_to_float(array, datetime_unit):
    if False:
        while True:
            i = 10
    'Convert numpy.timedelta64 to float.\n\n    Notes\n    -----\n    The array is first converted to microseconds, which is less likely to\n    cause overflow errors.\n    '
    array = array.astype('timedelta64[ns]').astype(np.float64)
    conversion_factor = np.timedelta64(1, 'ns') / np.timedelta64(1, datetime_unit)
    return conversion_factor * array

def pd_timedelta_to_float(value, datetime_unit):
    if False:
        for i in range(10):
            print('nop')
    'Convert pandas.Timedelta to float.\n\n    Notes\n    -----\n    Built on the assumption that pandas timedelta values are in nanoseconds,\n    which is also the numpy default resolution.\n    '
    value = value.to_timedelta64()
    return np_timedelta64_to_float(value, datetime_unit)

def _timedelta_to_seconds(array):
    if False:
        while True:
            i = 10
    if isinstance(array, datetime.timedelta):
        return array.total_seconds() * 1000000.0
    else:
        return np.reshape([a.total_seconds() for a in array.ravel()], array.shape) * 1000000.0

def py_timedelta_to_float(array, datetime_unit):
    if False:
        print('Hello World!')
    'Convert a timedelta object to a float, possibly at a loss of resolution.'
    array = asarray(array)
    if is_duck_dask_array(array):
        array = array.map_blocks(_timedelta_to_seconds, meta=np.array([], dtype=np.float64))
    else:
        array = _timedelta_to_seconds(array)
    conversion_factor = np.timedelta64(1, 'us') / np.timedelta64(1, datetime_unit)
    return conversion_factor * array

def mean(array, axis=None, skipna=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'inhouse mean that can handle np.datetime64 or cftime.datetime\n    dtypes'
    from xarray.core.common import _contains_cftime_datetimes
    array = asarray(array)
    if array.dtype.kind in 'Mm':
        offset = _datetime_nanmin(array)
        dtype = 'timedelta64[ns]'
        return _mean(datetime_to_numeric(array, offset), axis=axis, skipna=skipna, **kwargs).astype(dtype) + offset
    elif _contains_cftime_datetimes(array):
        offset = min(array)
        timedeltas = datetime_to_numeric(array, offset, datetime_unit='us')
        mean_timedeltas = _mean(timedeltas, axis=axis, skipna=skipna, **kwargs)
        return _to_pytimedelta(mean_timedeltas, unit='us') + offset
    else:
        return _mean(array, axis=axis, skipna=skipna, **kwargs)
mean.numeric_only = True

def _nd_cum_func(cum_func, array, axis, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    array = asarray(array)
    if axis is None:
        axis = tuple(range(array.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    out = array
    for ax in axis:
        out = cum_func(out, axis=ax, **kwargs)
    return out

def cumprod(array, axis=None, **kwargs):
    if False:
        return 10
    'N-dimensional version of cumprod.'
    return _nd_cum_func(cumprod_1d, array, axis, **kwargs)

def cumsum(array, axis=None, **kwargs):
    if False:
        print('Hello World!')
    'N-dimensional version of cumsum.'
    return _nd_cum_func(cumsum_1d, array, axis, **kwargs)

def first(values, axis, skipna=None):
    if False:
        while True:
            i = 10
    'Return the first non-NA elements in this array along the given axis'
    if (skipna or skipna is None) and values.dtype.kind not in 'iSU':
        if is_chunked_array(values):
            return chunked_nanfirst(values, axis)
        else:
            return nputils.nanfirst(values, axis)
    return take(values, 0, axis=axis)

def last(values, axis, skipna=None):
    if False:
        return 10
    'Return the last non-NA elements in this array along the given axis'
    if (skipna or skipna is None) and values.dtype.kind not in 'iSU':
        if is_chunked_array(values):
            return chunked_nanlast(values, axis)
        else:
            return nputils.nanlast(values, axis)
    return take(values, -1, axis=axis)

def least_squares(lhs, rhs, rcond=None, skipna=False):
    if False:
        return 10
    'Return the coefficients and residuals of a least-squares fit.'
    if is_duck_dask_array(rhs):
        return dask_array_ops.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
    else:
        return nputils.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)

def push(array, n, axis):
    if False:
        i = 10
        return i + 15
    from bottleneck import push
    if is_duck_dask_array(array):
        return dask_array_ops.push(array, n, axis)
    else:
        return push(array, n, axis)

def _first_last_wrapper(array, *, axis, op, keepdims):
    if False:
        for i in range(10):
            print('nop')
    return op(array, axis, keepdims=keepdims)

def _chunked_first_or_last(darray, axis, op):
    if False:
        return 10
    chunkmanager = get_chunked_array_type(darray)
    axis = normalize_axis_index(axis, darray.ndim)
    wrapped_op = partial(_first_last_wrapper, op=op)
    return chunkmanager.reduction(darray, func=wrapped_op, aggregate_func=wrapped_op, axis=axis, dtype=darray.dtype, keepdims=False)

def chunked_nanfirst(darray, axis):
    if False:
        print('Hello World!')
    return _chunked_first_or_last(darray, axis, op=nputils.nanfirst)

def chunked_nanlast(darray, axis):
    if False:
        while True:
            i = 10
    return _chunked_first_or_last(darray, axis, op=nputils.nanlast)