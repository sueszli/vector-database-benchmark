from typing import Union
import numpy as np
import talib
from jesse.helpers import np_shift
from jesse.helpers import slice_candles

def safezonestop(candles: np.ndarray, period: int=22, mult: float=2.5, max_lookback: int=3, direction: str='long', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    Safezone Stops\n\n    :param candles: np.ndarray\n    :param period: int - default: 22\n    :param mult: float - default: 2.5\n    :param max_lookback: int - default: 3\n    :param direction: str - default: long\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    high = candles[:, 3]
    low = candles[:, 4]
    last_high = np_shift(high, 1, fill_value=np.nan)
    last_low = np_shift(low, 1, fill_value=np.nan)
    if direction == 'long':
        res = talib.MAX(last_low - mult * talib.MINUS_DM(high, low, timeperiod=period), max_lookback)
    else:
        res = talib.MIN(last_high + mult * talib.PLUS_DM(high, low, timeperiod=period), max_lookback)
    return res if sequential else res[-1]