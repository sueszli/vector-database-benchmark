from __future__ import annotations
import ctypes
from typing import TYPE_CHECKING, Any
import numpy as np
if TYPE_CHECKING:
    from polars import Series

class SeriesView(np.ndarray):

    def __new__(cls, input_array: np.ndarray[Any, Any], owned_series: Series) -> SeriesView:
        if False:
            for i in range(10):
                print('nop')
        obj = input_array.view(cls)
        obj.owned_series = owned_series
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if False:
            i = 10
            return i + 15
        if obj is None:
            return
        self.owned_series = getattr(obj, 'owned_series', None)

def _ptr_to_numpy(ptr: int, len: int, ptr_type: Any) -> np.ndarray[Any, Any]:
    if False:
        print('Hello World!')
    '\n    Create a memory block view as a numpy array.\n\n    Parameters\n    ----------\n    ptr\n        C/Rust ptr casted to usize.\n    len\n        Length of the array values.\n    ptr_type\n        Example:\n            f32: ctypes.c_float)\n\n    Returns\n    -------\n    numpy.ndarray\n        View of memory block as numpy array.\n\n    '
    ptr_ctype = ctypes.cast(ptr, ctypes.POINTER(ptr_type))
    return np.ctypeslib.as_array(ptr_ctype, (len,))