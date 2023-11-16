from typing import Union
import numpy as np
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source, slice_candles

def edcf(candles: np.ndarray, period: int=15, source_type: str='hl2', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    Ehlers Distance Coefficient Filter\n\n    :param candles: np.ndarray\n    :param period: int - default: 15\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    res = edcf_fast(source, period)
    return res if sequential else res[-1]

@njit
def edcf_fast(source, period):
    if False:
        return 10
    newseries = np.full_like(source, np.nan)
    for j in range(2 * period, source.shape[0]):
        num = 0.0
        coefSum = 0.0
        for i in range(period):
            distance = 0.0
            for lb in range(1, period):
                distance += np.power(source[j - i] - source[j - i - lb], 2)
            num += distance * source[j - i]
            coefSum += distance
        newseries[j] = num / coefSum if coefSum != 0 else 0
    return newseries