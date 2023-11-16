from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def adosc(candles: np.ndarray, fast_period: int=3, slow_period: int=10, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    ADOSC - Chaikin A/D Oscillator\n\n    :param candles: np.ndarray\n    :param fast_period: int - default: 3\n    :param slow_period: int - default: 10\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = talib.ADOSC(candles[:, 3], candles[:, 4], candles[:, 2], candles[:, 5], fastperiod=fast_period, slowperiod=slow_period)
    return res if sequential else res[-1]