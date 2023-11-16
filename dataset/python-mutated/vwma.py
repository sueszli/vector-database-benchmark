from typing import Union
import numpy as np
import tulipy as ti
from jesse.helpers import get_candle_source, same_length
from jesse.helpers import slice_candles

def vwma(candles: np.ndarray, period: int=20, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    VWMA - Volume Weighted Moving Average\n\n    :param candles: np.ndarray\n    :param period: int - default: 20\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    res = ti.vwma(np.ascontiguousarray(source), np.ascontiguousarray(candles[:, 5]), period=period)
    return same_length(candles, res) if sequential else res[-1]