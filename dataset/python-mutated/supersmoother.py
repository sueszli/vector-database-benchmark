from typing import Union
import numpy as np
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source, slice_candles

def supersmoother(candles: np.ndarray, period: int=14, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    Super Smoother Filter 2pole Butterworth\n    This indicator was described by John F. Ehlers\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    res = supersmoother_fast(source, period)
    return res if sequential else res[-1]

@njit
def supersmoother_fast(source, period):
    if False:
        return 10
    a = np.exp(-1.414 * np.pi / period)
    b = 2 * a * np.cos(1.414 * np.pi / period)
    newseries = np.copy(source)
    for i in range(2, source.shape[0]):
        newseries[i] = (1 + a ** 2 - b) / 2 * (source[i] + source[i - 1]) + b * newseries[i - 1] - a ** 2 * newseries[i - 2]
    return newseries