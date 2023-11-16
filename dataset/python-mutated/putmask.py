"""
EA-compatible analogue to np.putmask
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import is_list_like
from pandas.core.arrays import ExtensionArray
if TYPE_CHECKING:
    from pandas._typing import ArrayLike, npt
    from pandas import MultiIndex

def putmask_inplace(values: ArrayLike, mask: npt.NDArray[np.bool_], value: Any) -> None:
    if False:
        return 10
    '\n    ExtensionArray-compatible implementation of np.putmask.  The main\n    difference is we do not handle repeating or truncating like numpy.\n\n    Parameters\n    ----------\n    values: np.ndarray or ExtensionArray\n    mask : np.ndarray[bool]\n        We assume extract_bool_array has already been called.\n    value : Any\n    '
    if not isinstance(values, np.ndarray) or (values.dtype == object and (not lib.is_scalar(value))) or (isinstance(value, np.ndarray) and (not np.can_cast(value.dtype, values.dtype))):
        if is_list_like(value) and len(value) == len(values):
            values[mask] = value[mask]
        else:
            values[mask] = value
    else:
        np.putmask(values, mask, value)

def putmask_without_repeat(values: np.ndarray, mask: npt.NDArray[np.bool_], new: Any) -> None:
    if False:
        while True:
            i = 10
    '\n    np.putmask will truncate or repeat if `new` is a listlike with\n    len(new) != len(values).  We require an exact match.\n\n    Parameters\n    ----------\n    values : np.ndarray\n    mask : np.ndarray[bool]\n    new : Any\n    '
    if getattr(new, 'ndim', 0) >= 1:
        new = new.astype(values.dtype, copy=False)
    nlocs = mask.sum()
    if nlocs > 0 and is_list_like(new) and (getattr(new, 'ndim', 1) == 1):
        shape = np.shape(new)
        if nlocs == shape[-1]:
            np.place(values, mask, new)
        elif mask.shape[-1] == shape[-1] or shape[-1] == 1:
            np.putmask(values, mask, new)
        else:
            raise ValueError('cannot assign mismatch length to masked array')
    else:
        np.putmask(values, mask, new)

def validate_putmask(values: ArrayLike | MultiIndex, mask: np.ndarray) -> tuple[npt.NDArray[np.bool_], bool]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Validate mask and check if this putmask operation is a no-op.\n    '
    mask = extract_bool_array(mask)
    if mask.shape != values.shape:
        raise ValueError('putmask: mask and data must be the same size')
    noop = not mask.any()
    return (mask, noop)

def extract_bool_array(mask: ArrayLike) -> npt.NDArray[np.bool_]:
    if False:
        while True:
            i = 10
    '\n    If we have a SparseArray or BooleanArray, convert it to ndarray[bool].\n    '
    if isinstance(mask, ExtensionArray):
        mask = mask.to_numpy(dtype=bool, na_value=False)
    mask = np.asarray(mask, dtype=bool)
    return mask

def setitem_datetimelike_compat(values: np.ndarray, num_set: int, other):
    if False:
        while True:
            i = 10
    '\n    Parameters\n    ----------\n    values : np.ndarray\n    num_set : int\n        For putmask, this is mask.sum()\n    other : Any\n    '
    if values.dtype == object:
        (dtype, _) = infer_dtype_from(other)
        if lib.is_np_dtype(dtype, 'mM'):
            if not is_list_like(other):
                other = [other] * num_set
            else:
                other = list(other)
    return other