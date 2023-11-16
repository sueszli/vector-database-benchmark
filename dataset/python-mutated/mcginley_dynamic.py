from typing import Union
import numpy as np
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source, slice_candles

def mcginley_dynamic(candles: np.ndarray, period: int=10, k: float=0.6, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    McGinley Dynamic\n\n    :param candles: np.ndarray\n    :param period: int - default: 10\n    :param k: float - default: 0.6\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    mg = md_fast(source, k, period)
    return mg if sequential else mg[-1]

@njit
def md_fast(source, k, period):
    if False:
        while True:
            i = 10
    mg = np.full_like(source, np.nan)
    for i in range(source.size):
        if i == 0:
            mg[i] = source[i]
        else:
            mg[i] = mg[i - 1] + (source[i] - mg[i - 1]) / max([k * period * (source[i] / mg[i - 1]) ** 4, 1])
    return mg