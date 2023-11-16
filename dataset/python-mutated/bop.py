from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def bop(candles: np.ndarray, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    BOP - Balance Of Power\n\n    :param candles: np.ndarray\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = talib.BOP(candles[:, 1], candles[:, 3], candles[:, 4], candles[:, 2])
    return res if sequential else res[-1]