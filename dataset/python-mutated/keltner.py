from collections import namedtuple
import numpy as np
import talib
from jesse.indicators.ma import ma
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles
KeltnerChannel = namedtuple('KeltnerChannel', ['upperband', 'middleband', 'lowerband'])

def keltner(candles: np.ndarray, period: int=20, multiplier: float=2, matype: int=1, source_type: str='close', sequential: bool=False) -> KeltnerChannel:
    if False:
        while True:
            i = 10
    '\n    Keltner Channels\n\n    :param candles: np.ndarray\n    :param period: int - default: 20\n    :param multiplier: float - default: 2\n    :param matype: int - default: 1\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: KeltnerChannel(upperband, middleband, lowerband)\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    e = ma(source, period=period, matype=matype, sequential=True)
    a = talib.ATR(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod=period)
    up = e + a * multiplier
    mid = e
    low = e - a * multiplier
    if sequential:
        return KeltnerChannel(up, mid, low)
    else:
        return KeltnerChannel(up[-1], mid[-1], low[-1])