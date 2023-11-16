from collections import namedtuple
import numpy as np
import talib
from jesse.helpers import slice_candles
AC = namedtuple('AC', ['osc', 'change'])

def acosc(candles: np.ndarray, sequential: bool=False) -> AC:
    if False:
        return 10
    '\n    Acceleration / Deceleration Oscillator (AC)\n\n    :param candles: np.ndarray\n    :param sequential: bool - default: False\n\n    :return: AC(osc, change)\n    '
    candles = slice_candles(candles, sequential)
    med = talib.MEDPRICE(candles[:, 3], candles[:, 4])
    ao = talib.SMA(med, 5) - talib.SMA(med, 34)
    res = ao - talib.SMA(ao, 5)
    mom = talib.MOM(res, timeperiod=1)
    if sequential:
        return AC(res, mom)
    else:
        return AC(res[-1], mom[-1])