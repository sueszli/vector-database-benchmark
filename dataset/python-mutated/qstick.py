from typing import Union
import numpy as np
import tulipy as ti
from jesse.helpers import slice_candles, same_length

def qstick(candles: np.ndarray, period: int=5, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Qstick\n\n    :param candles: np.ndarray\n    :param period: int - default: 5\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = ti.qstick(np.ascontiguousarray(candles[:, 1]), np.ascontiguousarray(candles[:, 2]), period=period)
    return same_length(candles, res) if sequential else res[-1]