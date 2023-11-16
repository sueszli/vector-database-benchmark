from typing import Union
import numpy as np
import tulipy as ti
from jesse.helpers import slice_candles, same_length

def emv(candles: np.ndarray, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    EMV - Ease of Movement\n\n    :param candles: np.ndarray\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = ti.emv(np.ascontiguousarray(candles[:, 3]), np.ascontiguousarray(candles[:, 4]), np.ascontiguousarray(candles[:, 5]))
    return same_length(candles, res) if sequential else res[-1]