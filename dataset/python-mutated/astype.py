"""
Functions for implementing 'astype' methods according to pandas conventions,
particularly ones that differ from numpy.
"""
from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, overload
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError
from pandas.core.dtypes.common import is_object_dtype, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype, NumpyEADtype
if TYPE_CHECKING:
    from pandas._typing import ArrayLike, DtypeObj, IgnoreRaise
    from pandas.core.arrays import ExtensionArray
_dtype_obj = np.dtype(object)

@overload
def _astype_nansafe(arr: np.ndarray, dtype: np.dtype, copy: bool=..., skipna: bool=...) -> np.ndarray:
    if False:
        print('Hello World!')
    ...

@overload
def _astype_nansafe(arr: np.ndarray, dtype: ExtensionDtype, copy: bool=..., skipna: bool=...) -> ExtensionArray:
    if False:
        i = 10
        return i + 15
    ...

def _astype_nansafe(arr: np.ndarray, dtype: DtypeObj, copy: bool=True, skipna: bool=False) -> ArrayLike:
    if False:
        i = 10
        return i + 15
    "\n    Cast the elements of an array to a given dtype a nan-safe manner.\n\n    Parameters\n    ----------\n    arr : ndarray\n    dtype : np.dtype or ExtensionDtype\n    copy : bool, default True\n        If False, a view will be attempted but may fail, if\n        e.g. the item sizes don't align.\n    skipna: bool, default False\n        Whether or not we should skip NaN when casting as a string-type.\n\n    Raises\n    ------\n    ValueError\n        The dtype was a datetime64/timedelta64 dtype, but it had no unit.\n    "
    if isinstance(dtype, ExtensionDtype):
        return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)
    elif not isinstance(dtype, np.dtype):
        raise ValueError('dtype must be np.dtype or ExtensionDtype')
    if arr.dtype.kind in 'mM':
        from pandas.core.construction import ensure_wrapped_if_datetimelike
        arr = ensure_wrapped_if_datetimelike(arr)
        res = arr.astype(dtype, copy=copy)
        return np.asarray(res)
    if issubclass(dtype.type, str):
        shape = arr.shape
        if arr.ndim > 1:
            arr = arr.ravel()
        return lib.ensure_string_array(arr, skipna=skipna, convert_na_value=False).reshape(shape)
    elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in 'iu':
        return _astype_float_to_int_nansafe(arr, dtype, copy)
    elif arr.dtype == object:
        if lib.is_np_dtype(dtype, 'M'):
            from pandas.core.arrays import DatetimeArray
            dta = DatetimeArray._from_sequence(arr, dtype=dtype)
            return dta._ndarray
        elif lib.is_np_dtype(dtype, 'm'):
            from pandas.core.construction import ensure_wrapped_if_datetimelike
            tdvals = array_to_timedelta64(arr).view('m8[ns]')
            tda = ensure_wrapped_if_datetimelike(tdvals)
            return tda.astype(dtype, copy=False)._ndarray
    if dtype.name in ('datetime64', 'timedelta64'):
        msg = f"The '{dtype.name}' dtype has no unit. Please pass in '{dtype.name}[ns]' instead."
        raise ValueError(msg)
    if copy or arr.dtype == object or dtype == object:
        return arr.astype(dtype, copy=True)
    return arr.astype(dtype, copy=copy)

def _astype_float_to_int_nansafe(values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    astype with a check preventing converting NaN to an meaningless integer value.\n    '
    if not np.isfinite(values).all():
        raise IntCastingNaNError('Cannot convert non-finite values (NA or inf) to integer')
    if dtype.kind == 'u':
        if not (values >= 0).all():
            raise ValueError(f'Cannot losslessly cast from {values.dtype} to {dtype}')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        return values.astype(dtype, copy=copy)

def astype_array(values: ArrayLike, dtype: DtypeObj, copy: bool=False) -> ArrayLike:
    if False:
        for i in range(10):
            print('nop')
    '\n    Cast array (ndarray or ExtensionArray) to the new dtype.\n\n    Parameters\n    ----------\n    values : ndarray or ExtensionArray\n    dtype : dtype object\n    copy : bool, default False\n        copy if indicated\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n    '
    if values.dtype == dtype:
        if copy:
            return values.copy()
        return values
    if not isinstance(values, np.ndarray):
        values = values.astype(dtype, copy=copy)
    else:
        values = _astype_nansafe(values, dtype, copy=copy)
    if isinstance(dtype, np.dtype) and issubclass(values.dtype.type, str):
        values = np.array(values, dtype=object)
    return values

def astype_array_safe(values: ArrayLike, dtype, copy: bool=False, errors: IgnoreRaise='raise') -> ArrayLike:
    if False:
        print('Hello World!')
    "\n    Cast array (ndarray or ExtensionArray) to the new dtype.\n\n    This basically is the implementation for DataFrame/Series.astype and\n    includes all custom logic for pandas (NaN-safety, converting str to object,\n    not allowing )\n\n    Parameters\n    ----------\n    values : ndarray or ExtensionArray\n    dtype : str, dtype convertible\n    copy : bool, default False\n        copy if indicated\n    errors : str, {'raise', 'ignore'}, default 'raise'\n        - ``raise`` : allow exceptions to be raised\n        - ``ignore`` : suppress exceptions. On error return original object\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n    "
    errors_legal_values = ('raise', 'ignore')
    if errors not in errors_legal_values:
        invalid_arg = f"Expected value of kwarg 'errors' to be one of {list(errors_legal_values)}. Supplied value is '{errors}'"
        raise ValueError(invalid_arg)
    if inspect.isclass(dtype) and issubclass(dtype, ExtensionDtype):
        msg = f"Expected an instance of {dtype.__name__}, but got the class instead. Try instantiating 'dtype'."
        raise TypeError(msg)
    dtype = pandas_dtype(dtype)
    if isinstance(dtype, NumpyEADtype):
        dtype = dtype.numpy_dtype
    try:
        new_values = astype_array(values, dtype, copy=copy)
    except (ValueError, TypeError):
        if errors == 'ignore':
            new_values = values
        else:
            raise
    return new_values

def astype_is_view(dtype: DtypeObj, new_dtype: DtypeObj) -> bool:
    if False:
        print('Hello World!')
    'Checks if astype avoided copying the data.\n\n    Parameters\n    ----------\n    dtype : Original dtype\n    new_dtype : target dtype\n\n    Returns\n    -------\n    True if new data is a view or not guaranteed to be a copy, False otherwise\n    '
    if isinstance(dtype, np.dtype) and (not isinstance(new_dtype, np.dtype)):
        (new_dtype, dtype) = (dtype, new_dtype)
    if dtype == new_dtype:
        return True
    elif isinstance(dtype, np.dtype) and isinstance(new_dtype, np.dtype):
        return False
    elif is_string_dtype(dtype) and is_string_dtype(new_dtype):
        return True
    elif is_object_dtype(dtype) and new_dtype.kind == 'O':
        return True
    elif dtype.kind in 'mM' and new_dtype.kind in 'mM':
        dtype = getattr(dtype, 'numpy_dtype', dtype)
        new_dtype = getattr(new_dtype, 'numpy_dtype', new_dtype)
        return getattr(dtype, 'unit', None) == getattr(new_dtype, 'unit', None)
    numpy_dtype = getattr(dtype, 'numpy_dtype', None)
    new_numpy_dtype = getattr(new_dtype, 'numpy_dtype', None)
    if numpy_dtype is None and isinstance(dtype, np.dtype):
        numpy_dtype = dtype
    if new_numpy_dtype is None and isinstance(new_dtype, np.dtype):
        new_numpy_dtype = new_dtype
    if numpy_dtype is not None and new_numpy_dtype is not None:
        return numpy_dtype == new_numpy_dtype
    return True