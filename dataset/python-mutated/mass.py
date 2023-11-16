from typing import Union
import numpy as np
import tulipy as ti
from jesse.helpers import slice_candles, same_length

def mass(candles: np.ndarray, period: int=5, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    MASS - Mass Index\n\n    :param candles: np.ndarray\n    :param period: int - default: 5\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = ti.mass(np.ascontiguousarray(candles[:, 3]), np.ascontiguousarray(candles[:, 4]), period=period)
    return same_length(candles, res) if sequential else res[-1]