from typing import Union
import numpy as np
import talib
from jesse.indicators.mean_ad import mean_ad
from jesse.indicators.median_ad import median_ad
from jesse.helpers import slice_candles

def devstop(candles: np.ndarray, period: int=20, mult: float=0, devtype: int=0, direction: str='long', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    Kase Dev Stops\n\n    :param candles: np.ndarray\n    :param period: int - default: 20\n    :param mult: float - default: 0\n    :param devtype: int - default: 0\n    :param direction: str - default: long\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    high = candles[:, 3]
    low = candles[:, 4]
    AVTR = talib.SMA(talib.MAX(high, 2) - talib.MIN(low, 2), period)
    if devtype == 0:
        SD = talib.STDDEV(talib.MAX(high, 2) - talib.MIN(low, 2), period)
    elif devtype == 1:
        SD = mean_ad(talib.MAX(high, 2) - talib.MIN(low, 2), period, sequential=True)
    elif devtype == 2:
        SD = median_ad(talib.MAX(high, 2) - talib.MIN(low, 2), period, sequential=True)
    if direction == 'long':
        res = talib.MAX(high - AVTR - mult * SD, period)
    else:
        res = talib.MIN(low + AVTR + mult * SD, period)
    return res if sequential else res[-1]