from typing import Union
import numpy as np
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source, slice_candles

def high_pass_2_pole(candles: np.ndarray, period: int=48, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        return 10
    '\n    (2 pole) high-pass filter indicator by John F. Ehlers\n\n    :param candles: np.ndarray\n    :param period: int - default: 48\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    hpf = high_pass_2_pole_fast(source, period)
    if sequential:
        return hpf
    else:
        return None if np.isnan(hpf[-1]) else hpf[-1]

@njit
def high_pass_2_pole_fast(source, period, K=0.707):
    if False:
        while True:
            i = 10
    alpha = 1 + (np.sin(2 * np.pi * K / period) - 1) / np.cos(2 * np.pi * K / period)
    newseries = np.copy(source)
    for i in range(2, source.shape[0]):
        newseries[i] = (1 - alpha / 2) ** 2 * source[i] - 2 * (1 - alpha / 2) ** 2 * source[i - 1] + (1 - alpha / 2) ** 2 * source[i - 2] + 2 * (1 - alpha) * newseries[i - 1] - (1 - alpha) ** 2 * newseries[i - 2]
    return newseries