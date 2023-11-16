"""
Utilities for working with numpy arrays.
"""
from collections import OrderedDict
from datetime import datetime
from distutils.version import StrictVersion
from warnings import catch_warnings, filterwarnings
import six
import numpy as np
from numpy import array_equal, broadcast, busday_count, datetime64, diff, dtype, empty, flatnonzero, hstack, isnan, nan, vectorize, where
from numpy.lib.stride_tricks import as_strided
from toolz import flip
numpy_version = StrictVersion(np.__version__)
uint8_dtype = dtype('uint8')
bool_dtype = dtype('bool')
uint32_dtype = dtype('uint32')
uint64_dtype = dtype('uint64')
int64_dtype = dtype('int64')
float32_dtype = dtype('float32')
float64_dtype = dtype('float64')
complex128_dtype = dtype('complex128')
datetime64D_dtype = dtype('datetime64[D]')
datetime64ns_dtype = dtype('datetime64[ns]')
object_dtype = dtype('O')
categorical_dtype = object_dtype
make_datetime64ns = flip(datetime64, 'ns')
make_datetime64D = flip(datetime64, 'D')
try:
    assert_array_compare = np.testing.utils.assert_array_compare
except AttributeError:
    assert_array_compare = np.testing.assert_array_compare
NaTmap = {dtype('datetime64[%s]' % unit): datetime64('NaT', unit) for unit in ('ns', 'us', 'ms', 's', 'm', 'D')}

def NaT_for_dtype(dtype):
    if False:
        return 10
    'Retrieve NaT with the same units as ``dtype``.\n\n    Parameters\n    ----------\n    dtype : dtype-coercable\n        The dtype to lookup the NaT value for.\n\n    Returns\n    -------\n    NaT : dtype\n        The NaT value for the given dtype.\n    '
    return NaTmap[np.dtype(dtype)]
NaTns = NaT_for_dtype(datetime64ns_dtype)
NaTD = NaT_for_dtype(datetime64D_dtype)
_FILLVALUE_DEFAULTS = {bool_dtype: False, float32_dtype: nan, float64_dtype: nan, datetime64ns_dtype: NaTns, object_dtype: None}
INT_DTYPES_BY_SIZE_BYTES = OrderedDict([(1, dtype('int8')), (2, dtype('int16')), (4, dtype('int32')), (8, dtype('int64'))])
UNSIGNED_INT_DTYPES_BY_SIZE_BYTES = OrderedDict([(1, dtype('uint8')), (2, dtype('uint16')), (4, dtype('uint32')), (8, dtype('uint64'))])

def int_dtype_with_size_in_bytes(size):
    if False:
        return 10
    try:
        return INT_DTYPES_BY_SIZE_BYTES[size]
    except KeyError:
        raise ValueError('No integral dtype whose size is %d bytes.' % size)

def unsigned_int_dtype_with_size_in_bytes(size):
    if False:
        i = 10
        return i + 15
    try:
        return UNSIGNED_INT_DTYPES_BY_SIZE_BYTES[size]
    except KeyError:
        raise ValueError('No unsigned integral dtype whose size is %d bytes.' % size)

class NoDefaultMissingValue(Exception):
    pass

def make_kind_check(python_types, numpy_kind):
    if False:
        i = 10
        return i + 15
    '\n    Make a function that checks whether a scalar or array is of a given kind\n    (e.g. float, int, datetime, timedelta).\n    '

    def check(value):
        if False:
            return 10
        if hasattr(value, 'dtype'):
            return value.dtype.kind == numpy_kind
        return isinstance(value, python_types)
    return check
is_float = make_kind_check(float, 'f')
is_int = make_kind_check(int, 'i')
is_datetime = make_kind_check(datetime, 'M')
is_object = make_kind_check(object, 'O')

def coerce_to_dtype(dtype, value):
    if False:
        print('Hello World!')
    '\n    Make a value with the specified numpy dtype.\n\n    Only datetime64[ns] and datetime64[D] are supported for datetime dtypes.\n    '
    name = dtype.name
    if name.startswith('datetime64'):
        if name == 'datetime64[D]':
            return make_datetime64D(value)
        elif name == 'datetime64[ns]':
            return make_datetime64ns(value)
        else:
            raise TypeError("Don't know how to coerce values of dtype %s" % dtype)
    return dtype.type(value)

def default_missing_value_for_dtype(dtype):
    if False:
        return 10
    '\n    Get the default fill value for `dtype`.\n    '
    try:
        return _FILLVALUE_DEFAULTS[dtype]
    except KeyError:
        raise NoDefaultMissingValue('No default value registered for dtype %s.' % dtype)

def repeat_first_axis(array, count):
    if False:
        print('Hello World!')
    '\n    Restride `array` to repeat `count` times along the first axis.\n\n    Parameters\n    ----------\n    array : np.array\n        The array to restride.\n    count : int\n        Number of times to repeat `array`.\n\n    Returns\n    -------\n    result : array\n        Array of shape (count,) + array.shape, composed of `array` repeated\n        `count` times along the first axis.\n\n    Example\n    -------\n    >>> from numpy import arange\n    >>> a = arange(3); a\n    array([0, 1, 2])\n    >>> repeat_first_axis(a, 2)\n    array([[0, 1, 2],\n           [0, 1, 2]])\n    >>> repeat_first_axis(a, 4)\n    array([[0, 1, 2],\n           [0, 1, 2],\n           [0, 1, 2],\n           [0, 1, 2]])\n\n    Notes\n    ----\n    The resulting array will share memory with `array`.  If you need to assign\n    to the input or output, you should probably make a copy first.\n\n    See Also\n    --------\n    repeat_last_axis\n    '
    return as_strided(array, (count,) + array.shape, (0,) + array.strides)

def repeat_last_axis(array, count):
    if False:
        return 10
    '\n    Restride `array` to repeat `count` times along the last axis.\n\n    Parameters\n    ----------\n    array : np.array\n        The array to restride.\n    count : int\n        Number of times to repeat `array`.\n\n    Returns\n    -------\n    result : array\n        Array of shape array.shape + (count,) composed of `array` repeated\n        `count` times along the last axis.\n\n    Example\n    -------\n    >>> from numpy import arange\n    >>> a = arange(3); a\n    array([0, 1, 2])\n    >>> repeat_last_axis(a, 2)\n    array([[0, 0],\n           [1, 1],\n           [2, 2]])\n    >>> repeat_last_axis(a, 4)\n    array([[0, 0, 0, 0],\n           [1, 1, 1, 1],\n           [2, 2, 2, 2]])\n\n    Notes\n    ----\n    The resulting array will share memory with `array`.  If you need to assign\n    to the input or output, you should probably make a copy first.\n\n    See Also\n    --------\n    repeat_last_axis\n    '
    return as_strided(array, array.shape + (count,), array.strides + (0,))

def rolling_window(array, length):
    if False:
        for i in range(10):
            print('nop')
    '\n    Restride an array of shape\n\n        (X_0, ... X_N)\n\n    into an array of shape\n\n        (length, X_0 - length + 1, ... X_N)\n\n    where each slice at index i along the first axis is equivalent to\n\n        result[i] = array[length * i:length * (i + 1)]\n\n    Parameters\n    ----------\n    array : np.ndarray\n        The base array.\n    length : int\n        Length of the synthetic first axis to generate.\n\n    Returns\n    -------\n    out : np.ndarray\n\n    Example\n    -------\n    >>> from numpy import arange\n    >>> a = arange(25).reshape(5, 5)\n    >>> a\n    array([[ 0,  1,  2,  3,  4],\n           [ 5,  6,  7,  8,  9],\n           [10, 11, 12, 13, 14],\n           [15, 16, 17, 18, 19],\n           [20, 21, 22, 23, 24]])\n\n    >>> rolling_window(a, 2)\n    array([[[ 0,  1,  2,  3,  4],\n            [ 5,  6,  7,  8,  9]],\n    <BLANKLINE>\n           [[ 5,  6,  7,  8,  9],\n            [10, 11, 12, 13, 14]],\n    <BLANKLINE>\n           [[10, 11, 12, 13, 14],\n            [15, 16, 17, 18, 19]],\n    <BLANKLINE>\n           [[15, 16, 17, 18, 19],\n            [20, 21, 22, 23, 24]]])\n    '
    orig_shape = array.shape
    if not orig_shape:
        raise IndexError("Can't restride a scalar.")
    elif orig_shape[0] <= length:
        raise IndexError("Can't restride array of shape {shape} with a window length of {len}".format(shape=orig_shape, len=length))
    num_windows = orig_shape[0] - length + 1
    new_shape = (num_windows, length) + orig_shape[1:]
    new_strides = (array.strides[0],) + array.strides
    return as_strided(array, new_shape, new_strides)
_notNaT = make_datetime64D(0)
iNaT = int(NaTns.view(int64_dtype))
assert iNaT == NaTD.view(int64_dtype), 'iNaTns != iNaTD'

def isnat(obj):
    if False:
        print('Hello World!')
    '\n    Check if a value is np.NaT.\n    '
    if obj.dtype.kind not in ('m', 'M'):
        raise ValueError('%s is not a numpy datetime or timedelta')
    return obj.view(int64_dtype) == iNaT

def is_missing(data, missing_value):
    if False:
        while True:
            i = 10
    '\n    Generic is_missing function that handles NaN and NaT.\n    '
    if is_float(data) and isnan(missing_value):
        return isnan(data)
    elif is_datetime(data) and isnat(missing_value):
        return isnat(data)
    elif is_object(data) and missing_value is None:
        return data == np.array([missing_value])
    return data == missing_value

def same(x, y):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if two scalar values are "the same".\n\n    Returns True if `x == y`, or if x and y are both NaN or both NaT.\n    '
    if is_float(x) and isnan(x) and is_float(y) and isnan(y):
        return True
    elif is_datetime(x) and isnat(x) and is_datetime(y) and isnat(y):
        return True
    else:
        return x == y

def busday_count_mask_NaT(begindates, enddates, out=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Simple of numpy.busday_count that returns `float` arrays rather than int\n    arrays, and handles `NaT`s by returning `NaN`s where the inputs were `NaT`.\n\n    Doesn't support custom weekdays or calendars, but probably should in the\n    future.\n\n    See Also\n    --------\n    np.busday_count\n    "
    if out is None:
        out = empty(broadcast(begindates, enddates).shape, dtype=float)
    beginmask = isnat(begindates)
    endmask = isnat(enddates)
    out = busday_count(where(beginmask, _notNaT, begindates), where(endmask, _notNaT, enddates), out=out)
    out[beginmask | endmask] = nan
    return out

class WarningContext(object):
    """
    Re-usable contextmanager for contextually managing warnings.
    """

    def __init__(self, *warning_specs):
        if False:
            print('Hello World!')
        self._warning_specs = warning_specs
        self._catchers = []

    def __enter__(self):
        if False:
            while True:
                i = 10
        catcher = catch_warnings()
        catcher.__enter__()
        self._catchers.append(catcher)
        for (args, kwargs) in self._warning_specs:
            filterwarnings(*args, **kwargs)
        return self

    def __exit__(self, *exc_info):
        if False:
            i = 10
            return i + 15
        catcher = self._catchers.pop()
        return catcher.__exit__(*exc_info)

def ignore_nanwarnings():
    if False:
        for i in range(10):
            print('nop')
    "\n    Helper for building a WarningContext that ignores warnings from numpy's\n    nanfunctions.\n    "
    return WarningContext((('ignore',), {'category': RuntimeWarning, 'module': 'numpy.lib.nanfunctions'}))

def vectorized_is_element(array, choices):
    if False:
        while True:
            i = 10
    '\n    Check if each element of ``array`` is in choices.\n\n    Parameters\n    ----------\n    array : np.ndarray\n    choices : object\n        Object implementing __contains__.\n\n    Returns\n    -------\n    was_element : np.ndarray[bool]\n        Array indicating whether each element of ``array`` was in ``choices``.\n    '
    return vectorize(choices.__contains__, otypes=[bool])(array)

def as_column(a):
    if False:
        return 10
    '\n    Convert an array of shape (N,) into an array of shape (N, 1).\n\n    This is equivalent to `a[:, np.newaxis]`.\n\n    Parameters\n    ----------\n    a : np.ndarray\n\n    Example\n    -------\n    >>> import numpy as np\n    >>> a = np.arange(5)\n    >>> a\n    array([0, 1, 2, 3, 4])\n    >>> as_column(a)\n    array([[0],\n           [1],\n           [2],\n           [3],\n           [4]])\n    >>> as_column(a).shape\n    (5, 1)\n    '
    if a.ndim != 1:
        raise ValueError('as_column expected an 1-dimensional array, but got an array of shape %s' % (a.shape,))
    return a[:, None]

def changed_locations(a, include_first):
    if False:
        i = 10
        return i + 15
    '\n    Compute indices of values in ``a`` that differ from the previous value.\n\n    Parameters\n    ----------\n    a : np.ndarray\n        The array on which to indices of change.\n    include_first : bool\n        Whether or not to consider the first index of the array as "changed".\n\n    Example\n    -------\n    >>> import numpy as np\n    >>> changed_locations(np.array([0, 0, 5, 5, 1, 1]), include_first=False)\n    array([2, 4])\n\n    >>> changed_locations(np.array([0, 0, 5, 5, 1, 1]), include_first=True)\n    array([0, 2, 4])\n    '
    if a.ndim > 1:
        raise ValueError('indices_of_changed_values only supports 1D arrays.')
    indices = flatnonzero(diff(a)) + 1
    if not include_first:
        return indices
    return hstack([[0], indices])

def compare_datetime_arrays(x, y):
    if False:
        print('Hello World!')
    '\n    Compare datetime64 ndarrays, treating NaT values as equal.\n    '
    return array_equal(x.view('int64'), y.view('int64'))

def bytes_array_to_native_str_object_array(a):
    if False:
        i = 10
        return i + 15
    'Convert an array of dtype S to an object array containing `str`.\n    '
    if six.PY2:
        return a.astype(object)
    else:
        return a.astype(str).astype(object)