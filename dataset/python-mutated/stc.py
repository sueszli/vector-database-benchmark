from typing import Union
import numpy as np
import talib
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles
from jesse.indicators.ma import ma

def stc(candles: np.ndarray, fast_period: int=23, fast_matype: int=1, slow_period: int=50, slow_matype: int=1, k_period: int=10, d_period: int=3, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    STC - Schaff Trend Cycle (Oscillator)\n\n    :param candles: np.ndarray\n    :param fast_period: int - default: 23\n    :param fast_matype: int - default: 1\n    :param slow_period: int - default: 50\n    :param slow_matype: int - default: 1\n    :param k_period: int - default: 10\n    :param d_period: int - default: 3\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    macd = ma(source, period=fast_period, matype=fast_matype, sequential=True) - ma(source, period=slow_period, matype=slow_matype, sequential=True)
    stok = (macd - talib.MIN(macd, k_period)) / (talib.MAX(macd, k_period) - talib.MIN(macd, k_period)) * 100
    d = talib.EMA(stok, d_period)
    kd = (d - talib.MIN(d, k_period)) / (talib.MAX(d, k_period) - talib.MIN(d, k_period)) * 100
    res = talib.EMA(kd, d_period)
    return res if sequential else res[-1]