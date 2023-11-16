from typing import Union
import numpy as np
from jesse.indicators.ma import ma
from jesse.helpers import slice_candles

def kaufmanstop(candles: np.ndarray, period: int=22, mult: float=2, direction: str='long', matype: int=0, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Perry Kaufman's Stops\n\n    :param candles: np.ndarray\n    :param period: int - default: 22\n    :param mult: float - default: 2\n    :param direction: str - default: long\n    :param matype: int - default: 0\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    "
    candles = slice_candles(candles, sequential)
    high = candles[:, 3]
    low = candles[:, 4]
    hl_diff = ma(high - low, period=period, matype=matype, sequential=True)
    res = low - hl_diff * mult if direction == 'long' else high + hl_diff * mult
    return res if sequential else res[-1]