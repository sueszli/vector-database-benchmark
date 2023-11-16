from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from rerun.color_conversion import u8_array_to_rgba
if TYPE_CHECKING:
    from . import Rgba32ArrayLike, Rgba32Like

def _numpy_array_to_u32(data: npt.NDArray[np.uint8 | np.float32 | np.float64]) -> npt.NDArray[np.uint32]:
    if False:
        print('Hello World!')
    if data.size == 0:
        return np.array([], dtype=np.uint32)
    if data.dtype.type in [np.float32, np.float64]:
        array = u8_array_to_rgba(np.asarray(np.round(np.asarray(data) * 255.0), np.uint8))
    elif data.dtype.type == np.uint32:
        array = np.asarray(data, dtype=np.uint32).flatten()
    else:
        array = u8_array_to_rgba(np.asarray(data, dtype=np.uint8))
    return array

class Rgba32Ext:
    """Extension for [Rgba32][rerun.datatypes.Rgba32]."""
    '\n    Extension for the `Rgba32` datatype.\n\n    Possible input for `Rgba32`:\n    - Sequence[int]: interpreted as rgb or rgba values in 0-255 range\n    - numpy array: interpreted as rgb or rgba values, range depending on dtype\n    - anything else (int or convertible to int): interpreted as a 32-bit packed rgba value\n\n    Possible inputs for `Rgba32Batch()`:\n    - a single `Rgba32` instance\n    - a sequence of `Rgba32` instances\n    - Nx3 or Nx4 numpy array, range depending on dtype\n    '

    @staticmethod
    def rgba__field_converter_override(data: Rgba32Like) -> int:
        if False:
            return 10
        from . import Rgba32
        if isinstance(data, Rgba32):
            return data.rgba
        if isinstance(data, np.ndarray):
            return int(_numpy_array_to_u32(data.reshape((1, -1)))[0])
        elif isinstance(data, Sequence):
            data = np.array(data).reshape((1, -1))
            if data.shape[1] not in (3, 4):
                raise ValueError(f'expected sequence of length of 3 or 4, received {data.shape[1]}')
            return int(_numpy_array_to_u32(data)[0])
        else:
            return int(data)

    @staticmethod
    def native_to_pa_array_override(data: Rgba32ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            while True:
                i = 10
        from . import Rgba32
        if isinstance(data, int) or isinstance(data, Rgba32):
            int_array = np.array([data])
        elif isinstance(data, Sequence) and len(data) == 0:
            int_array = np.array([])
        else:
            try:
                arr = np.asarray(data)
                if arr.dtype == np.uint32:
                    int_array = arr.flatten()
                else:
                    if len(arr.shape) == 1:
                        if arr.size > 4:
                            arr = arr.reshape((-1, 4))
                        else:
                            arr = arr.reshape((1, -1))
                    int_array = _numpy_array_to_u32(arr)
            except (ValueError, TypeError, IndexError):
                data_list = list(data)
                try:
                    data_list = [Rgba32(data_list)]
                except (IndexError, ValueError):
                    pass
                int_array = np.array([Rgba32(datum) for datum in data_list], np.uint32)
        return pa.array(int_array, type=data_type)