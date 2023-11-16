from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def aroonosc(candles: np.ndarray, period: int=14, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    AROONOSC - Aroon Oscillator\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = talib.AROONOSC(candles[:, 3], candles[:, 4], timeperiod=period)
    return res if sequential else res[-1]