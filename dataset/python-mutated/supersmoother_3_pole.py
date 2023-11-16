from typing import Union
import numpy as np
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source, slice_candles

def supersmoother_3_pole(candles: np.ndarray, period: int=14, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    Super Smoother Filter 3pole Butterworth\n    This indicator was described by John F. Ehlers\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
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
    a = np.exp(-np.pi / period)
    b = 2 * a * np.cos(1.738 * np.pi / period)
    c = a ** 2
    newseries = np.copy(source)
    for i in range(3, source.shape[0]):
        newseries[i] = (1 - c ** 2 - b + b * c) * source[i] + (b + c) * newseries[i - 1] + (-c - b * c) * newseries[i - 2] + c ** 2 * newseries[i - 3]
    return newseries