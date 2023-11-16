from __future__ import annotations
import warnings
import numpy as np
import pyarrow
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level

def fallback_performancewarning(version: str | None=None) -> None:
    if False:
        while True:
            i = 10
    "\n    Raise a PerformanceWarning for falling back to ExtensionArray's\n    non-pyarrow method\n    "
    msg = 'Falling back on a non-pyarrow code path which may decrease performance.'
    if version is not None:
        msg += f' Upgrade to pyarrow >={version} to possibly suppress this warning.'
    warnings.warn(msg, PerformanceWarning, stacklevel=find_stack_level())

def pyarrow_array_to_numpy_and_mask(arr, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    if False:
        return 10
    '\n    Convert a primitive pyarrow.Array to a numpy array and boolean mask based\n    on the buffers of the Array.\n\n    At the moment pyarrow.BooleanArray is not supported.\n\n    Parameters\n    ----------\n    arr : pyarrow.Array\n    dtype : numpy.dtype\n\n    Returns\n    -------\n    (data, mask)\n        Tuple of two numpy arrays with the raw data (with specified dtype) and\n        a boolean mask (validity mask, so False means missing)\n    '
    dtype = np.dtype(dtype)
    if pyarrow.types.is_null(arr.type):
        data = np.empty(len(arr), dtype=dtype)
        mask = np.zeros(len(arr), dtype=bool)
        return (data, mask)
    buflist = arr.buffers()
    offset = arr.offset * dtype.itemsize
    length = len(arr) * dtype.itemsize
    data_buf = buflist[1][offset:offset + length]
    data = np.frombuffer(data_buf, dtype=dtype)
    bitmask = buflist[0]
    if bitmask is not None:
        mask = pyarrow.BooleanArray.from_buffers(pyarrow.bool_(), len(arr), [None, bitmask], offset=arr.offset)
        mask = np.asarray(mask)
    else:
        mask = np.ones(len(arr), dtype=bool)
    return (data, mask)