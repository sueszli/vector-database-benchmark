from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def ultosc(candles: np.ndarray, timeperiod1: int=7, timeperiod2: int=14, timeperiod3: int=28, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    ULTOSC - Ultimate Oscillator\n\n    :param candles: np.ndarray\n    :param timeperiod1: int - default: 7\n    :param timeperiod2: int - default: 14\n    :param timeperiod3: int - default: 28\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = talib.ULTOSC(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
    return res if sequential else res[-1]