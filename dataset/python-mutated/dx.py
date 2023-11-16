from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def dx(candles: np.ndarray, period: int=14, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    DX - Directional Movement Index\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = talib.DX(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod=period)
    return res if sequential else res[-1]