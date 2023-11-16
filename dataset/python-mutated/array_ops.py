"""
Functions for arithmetic and comparison operations on NumPy arrays and
ExtensionArrays.
"""
from __future__ import annotations
import datetime
from functools import partial
import operator
from typing import TYPE_CHECKING, Any
import warnings
import numpy as np
from pandas._libs import NaT, Timedelta, Timestamp, lib, ops as libops
from pandas._libs.tslibs import BaseOffset, get_supported_reso, get_unit_from_dtype, is_supported_unit, is_unitless, npy_unit_to_abbrev
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike, find_common_type
from pandas.core.dtypes.common import ensure_object, is_bool_dtype, is_list_like, is_numeric_v_string_like, is_object_dtype, is_scalar
from pandas.core.dtypes.generic import ABCExtensionArray, ABCIndex, ABCSeries
from pandas.core.dtypes.missing import isna, notna
from pandas.core import roperator
from pandas.core.computation import expressions
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.ops import missing
from pandas.core.ops.dispatch import should_extension_dispatch
from pandas.core.ops.invalid import invalid_comparison
if TYPE_CHECKING:
    from pandas._typing import ArrayLike, Shape

def fill_binop(left, right, fill_value):
    if False:
        for i in range(10):
            print('nop')
    '\n    If a non-None fill_value is given, replace null entries in left and right\n    with this value, but only in positions where _one_ of left/right is null,\n    not both.\n\n    Parameters\n    ----------\n    left : array-like\n    right : array-like\n    fill_value : object\n\n    Returns\n    -------\n    left : array-like\n    right : array-like\n\n    Notes\n    -----\n    Makes copies if fill_value is not None and NAs are present.\n    '
    if fill_value is not None:
        left_mask = isna(left)
        right_mask = isna(right)
        mask = left_mask ^ right_mask
        if left_mask.any():
            left = left.copy()
            left[left_mask & mask] = fill_value
        if right_mask.any():
            right = right.copy()
            right[right_mask & mask] = fill_value
    return (left, right)

def comp_method_OBJECT_ARRAY(op, x, y):
    if False:
        i = 10
        return i + 15
    if isinstance(y, list):
        y = construct_1d_object_array_from_listlike(y)
    if isinstance(y, (np.ndarray, ABCSeries, ABCIndex)):
        if not is_object_dtype(y.dtype):
            y = y.astype(np.object_)
        if isinstance(y, (ABCSeries, ABCIndex)):
            y = y._values
        if x.shape != y.shape:
            raise ValueError('Shapes must match', x.shape, y.shape)
        result = libops.vec_compare(x.ravel(), y.ravel(), op)
    else:
        result = libops.scalar_compare(x.ravel(), y, op)
    return result.reshape(x.shape)

def _masked_arith_op(x: np.ndarray, y, op):
    if False:
        for i in range(10):
            print('nop')
    '\n    If the given arithmetic operation fails, attempt it again on\n    only the non-null elements of the input array(s).\n\n    Parameters\n    ----------\n    x : np.ndarray\n    y : np.ndarray, Series, Index\n    op : binary operator\n    '
    xrav = x.ravel()
    if isinstance(y, np.ndarray):
        dtype = find_common_type([x.dtype, y.dtype])
        result = np.empty(x.size, dtype=dtype)
        if len(x) != len(y):
            raise ValueError(x.shape, y.shape)
        ymask = notna(y)
        yrav = y.ravel()
        mask = notna(xrav) & ymask.ravel()
        if mask.any():
            result[mask] = op(xrav[mask], yrav[mask])
    else:
        if not is_scalar(y):
            raise TypeError(f'Cannot broadcast np.ndarray with operand of type {type(y)}')
        result = np.empty(x.size, dtype=x.dtype)
        mask = notna(xrav)
        if op is pow:
            mask = np.where(x == 1, False, mask)
        elif op is roperator.rpow:
            mask = np.where(y == 1, False, mask)
        if mask.any():
            result[mask] = op(xrav[mask], y)
    np.putmask(result, ~mask, np.nan)
    result = result.reshape(x.shape)
    return result

def _na_arithmetic_op(left: np.ndarray, right, op, is_cmp: bool=False):
    if False:
        return 10
    '\n    Return the result of evaluating op on the passed in values.\n\n    If native types are not compatible, try coercion to object dtype.\n\n    Parameters\n    ----------\n    left : np.ndarray\n    right : np.ndarray or scalar\n        Excludes DataFrame, Series, Index, ExtensionArray.\n    is_cmp : bool, default False\n        If this a comparison operation.\n\n    Returns\n    -------\n    array-like\n\n    Raises\n    ------\n    TypeError : invalid operation\n    '
    if isinstance(right, str):
        func = op
    else:
        func = partial(expressions.evaluate, op)
    try:
        result = func(left, right)
    except TypeError:
        if not is_cmp and (left.dtype == object or getattr(right, 'dtype', None) == object):
            result = _masked_arith_op(left, right, op)
        else:
            raise
    if is_cmp and (is_scalar(result) or result is NotImplemented):
        return invalid_comparison(left, right, op)
    return missing.dispatch_fill_zeros(op, left, right, result)

def arithmetic_op(left: ArrayLike, right: Any, op):
    if False:
        return 10
    '\n    Evaluate an arithmetic operation `+`, `-`, `*`, `/`, `//`, `%`, `**`, ...\n\n    Note: the caller is responsible for ensuring that numpy warnings are\n    suppressed (with np.errstate(all="ignore")) if needed.\n\n    Parameters\n    ----------\n    left : np.ndarray or ExtensionArray\n    right : object\n        Cannot be a DataFrame or Index.  Series is *not* excluded.\n    op : {operator.add, operator.sub, ...}\n        Or one of the reversed variants from roperator.\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n        Or a 2-tuple of these in the case of divmod or rdivmod.\n    '
    if should_extension_dispatch(left, right) or isinstance(right, (Timedelta, BaseOffset, Timestamp)) or right is NaT:
        res_values = op(left, right)
    else:
        _bool_arith_check(op, left, right)
        res_values = _na_arithmetic_op(left, right, op)
    return res_values

def comparison_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    if False:
        i = 10
        return i + 15
    '\n    Evaluate a comparison operation `=`, `!=`, `>=`, `>`, `<=`, or `<`.\n\n    Note: the caller is responsible for ensuring that numpy warnings are\n    suppressed (with np.errstate(all="ignore")) if needed.\n\n    Parameters\n    ----------\n    left : np.ndarray or ExtensionArray\n    right : object\n        Cannot be a DataFrame, Series, or Index.\n    op : {operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le}\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n    '
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = ensure_wrapped_if_datetimelike(right)
    rvalues = lib.item_from_zerodim(rvalues)
    if isinstance(rvalues, list):
        rvalues = np.asarray(rvalues)
    if isinstance(rvalues, (np.ndarray, ABCExtensionArray)):
        if len(lvalues) != len(rvalues):
            raise ValueError('Lengths must match to compare', lvalues.shape, rvalues.shape)
    if should_extension_dispatch(lvalues, rvalues) or ((isinstance(rvalues, (Timedelta, BaseOffset, Timestamp)) or right is NaT) and lvalues.dtype != object):
        res_values = op(lvalues, rvalues)
    elif is_scalar(rvalues) and isna(rvalues):
        if op is operator.ne:
            res_values = np.ones(lvalues.shape, dtype=bool)
        else:
            res_values = np.zeros(lvalues.shape, dtype=bool)
    elif is_numeric_v_string_like(lvalues, rvalues):
        return invalid_comparison(lvalues, rvalues, op)
    elif lvalues.dtype == object or isinstance(rvalues, str):
        res_values = comp_method_OBJECT_ARRAY(op, lvalues, rvalues)
    else:
        res_values = _na_arithmetic_op(lvalues, rvalues, op, is_cmp=True)
    return res_values

def na_logical_op(x: np.ndarray, y, op):
    if False:
        return 10
    try:
        result = op(x, y)
    except TypeError:
        if isinstance(y, np.ndarray):
            assert not (x.dtype.kind == 'b' and y.dtype.kind == 'b')
            x = ensure_object(x)
            y = ensure_object(y)
            result = libops.vec_binop(x.ravel(), y.ravel(), op)
        else:
            assert lib.is_scalar(y)
            if not isna(y):
                y = bool(y)
            try:
                result = libops.scalar_binop(x, y, op)
            except (TypeError, ValueError, AttributeError, OverflowError, NotImplementedError) as err:
                typ = type(y).__name__
                raise TypeError(f"Cannot perform '{op.__name__}' with a dtyped [{x.dtype}] array and scalar of type [{typ}]") from err
    return result.reshape(x.shape)

def logical_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    if False:
        return 10
    '\n    Evaluate a logical operation `|`, `&`, or `^`.\n\n    Parameters\n    ----------\n    left : np.ndarray or ExtensionArray\n    right : object\n        Cannot be a DataFrame, Series, or Index.\n    op : {operator.and_, operator.or_, operator.xor}\n        Or one of the reversed variants from roperator.\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n    '

    def fill_bool(x, left=None):
        if False:
            for i in range(10):
                print('nop')
        if x.dtype.kind in 'cfO':
            mask = isna(x)
            if mask.any():
                x = x.astype(object)
                x[mask] = False
        if left is None or left.dtype.kind == 'b':
            x = x.astype(bool)
        return x
    right = lib.item_from_zerodim(right)
    if is_list_like(right) and (not hasattr(right, 'dtype')):
        warnings.warn('Logical ops (and, or, xor) between Pandas objects and dtype-less sequences (e.g. list, tuple) are deprecated and will raise in a future version. Wrap the object in a Series, Index, or np.array before operating instead.', FutureWarning, stacklevel=find_stack_level())
        right = construct_1d_object_array_from_listlike(right)
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = right
    if should_extension_dispatch(lvalues, rvalues):
        res_values = op(lvalues, rvalues)
    else:
        if isinstance(rvalues, np.ndarray):
            is_other_int_dtype = rvalues.dtype.kind in 'iu'
            if not is_other_int_dtype:
                rvalues = fill_bool(rvalues, lvalues)
        else:
            is_other_int_dtype = lib.is_integer(rvalues)
        res_values = na_logical_op(lvalues, rvalues, op)
        if not (left.dtype.kind in 'iu' and is_other_int_dtype):
            res_values = fill_bool(res_values)
    return res_values

def get_array_op(op):
    if False:
        i = 10
        return i + 15
    '\n    Return a binary array operation corresponding to the given operator op.\n\n    Parameters\n    ----------\n    op : function\n        Binary operator from operator or roperator module.\n\n    Returns\n    -------\n    functools.partial\n    '
    if isinstance(op, partial):
        return op
    op_name = op.__name__.strip('_').lstrip('r')
    if op_name == 'arith_op':
        return op
    if op_name in {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}:
        return partial(comparison_op, op=op)
    elif op_name in {'and', 'or', 'xor', 'rand', 'ror', 'rxor'}:
        return partial(logical_op, op=op)
    elif op_name in {'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow'}:
        return partial(arithmetic_op, op=op)
    else:
        raise NotImplementedError(op_name)

def maybe_prepare_scalar_for_op(obj, shape: Shape):
    if False:
        print('Hello World!')
    '\n    Cast non-pandas objects to pandas types to unify behavior of arithmetic\n    and comparison operations.\n\n    Parameters\n    ----------\n    obj: object\n    shape : tuple[int]\n\n    Returns\n    -------\n    out : object\n\n    Notes\n    -----\n    Be careful to call this *after* determining the `name` attribute to be\n    attached to the result of the arithmetic operation.\n    '
    if type(obj) is datetime.timedelta:
        return Timedelta(obj)
    elif type(obj) is datetime.datetime:
        return Timestamp(obj)
    elif isinstance(obj, np.datetime64):
        if isna(obj):
            from pandas.core.arrays import DatetimeArray
            if is_unitless(obj.dtype):
                obj = obj.astype('datetime64[ns]')
            elif not is_supported_unit(get_unit_from_dtype(obj.dtype)):
                unit = get_unit_from_dtype(obj.dtype)
                closest_unit = npy_unit_to_abbrev(get_supported_reso(unit))
                obj = obj.astype(f'datetime64[{closest_unit}]')
            right = np.broadcast_to(obj, shape)
            return DatetimeArray(right)
        return Timestamp(obj)
    elif isinstance(obj, np.timedelta64):
        if isna(obj):
            from pandas.core.arrays import TimedeltaArray
            if is_unitless(obj.dtype):
                obj = obj.astype('timedelta64[ns]')
            elif not is_supported_unit(get_unit_from_dtype(obj.dtype)):
                unit = get_unit_from_dtype(obj.dtype)
                closest_unit = npy_unit_to_abbrev(get_supported_reso(unit))
                obj = obj.astype(f'timedelta64[{closest_unit}]')
            right = np.broadcast_to(obj, shape)
            return TimedeltaArray(right)
        return Timedelta(obj)
    return obj
_BOOL_OP_NOT_ALLOWED = {operator.truediv, roperator.rtruediv, operator.floordiv, roperator.rfloordiv, operator.pow, roperator.rpow}

def _bool_arith_check(op, a: np.ndarray, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    In contrast to numpy, pandas raises an error for certain operations\n    with booleans.\n    '
    if op in _BOOL_OP_NOT_ALLOWED:
        if a.dtype.kind == 'b' and (is_bool_dtype(b) or lib.is_bool(b)):
            op_name = op.__name__.strip('_').lstrip('r')
            raise NotImplementedError(f"operator '{op_name}' not implemented for bool dtypes")