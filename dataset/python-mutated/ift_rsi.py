from typing import Union
import talib
import numpy as np
from jesse.helpers import get_candle_source, slice_candles, same_length

def ift_rsi(candles: np.ndarray, rsi_period: int=5, wma_period: int=9, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    Modified Inverse Fisher Transform applied on RSI\n\n    :param candles: np.ndarray\n    :param rsi_period: int - default: 5\n    :param wma_period: int - default: 9\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    v1 = 0.1 * (talib.RSI(source, rsi_period) - 50)
    v2 = talib.WMA(v1, wma_period)
    res = ((2 * v2) ** 2 - 1) / ((2 * v2) ** 2 + 1)
    return same_length(candles, res) if sequential else res[-1]