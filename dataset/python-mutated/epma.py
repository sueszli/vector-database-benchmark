from typing import Union
import numpy as np
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source, slice_candles

def epma(candles: np.ndarray, period: int=11, offset: int=4, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    End Point Moving Average\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param offset: int - default: 4\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    res = epma_fast(source, period, offset)
    return res if sequential else res[-1]

@njit
def epma_fast(source, period, offset):
    if False:
        i = 10
        return i + 15
    newseries = np.copy(source)
    for j in range(period + offset + 1, source.shape[0]):
        my_sum = 0.0
        weightSum = 0.0
        for i in range(period - 1):
            weight = period - i - offset
            my_sum += source[j - i] * weight
            weightSum += weight
        newseries[j] = 1 / weightSum * my_sum
    return newseries