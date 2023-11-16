from collections import namedtuple
import numpy as np
import talib
from jesse.helpers import slice_candles
DM = namedtuple('DM', ['plus', 'minus'])

def dm(candles: np.ndarray, period: int=14, sequential: bool=False) -> DM:
    if False:
        while True:
            i = 10
    '\n    DM - Directional Movement\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param sequential: bool - default: False\n\n    :return: DM(plus, minus)\n    '
    candles = slice_candles(candles, sequential)
    MINUS_DI = talib.MINUS_DM(candles[:, 3], candles[:, 4], timeperiod=period)
    PLUS_DI = talib.PLUS_DM(candles[:, 3], candles[:, 4], timeperiod=period)
    if sequential:
        return DM(PLUS_DI, MINUS_DI)
    else:
        return DM(PLUS_DI[-1], MINUS_DI[-1])