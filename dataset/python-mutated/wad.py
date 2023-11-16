from typing import Union
import numpy as np
import tulipy as ti
from jesse.helpers import slice_candles, same_length

def wad(candles: np.ndarray, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    WAD - Williams Accumulation/Distribution\n\n    :param candles: np.ndarray\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = ti.wad(np.ascontiguousarray(candles[:, 3]), np.ascontiguousarray(candles[:, 4]), np.ascontiguousarray(candles[:, 1]))
    return same_length(candles, res) if sequential else res[-1]