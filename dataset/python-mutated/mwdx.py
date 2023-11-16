from typing import Union
import numpy as np
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source, slice_candles

def mwdx(candles: np.ndarray, factor: float=0.2, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        return 10
    '\n    MWDX Average\n\n    :param candles: np.ndarray\n    :param factor: float - default: 0.2\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    val2 = 2 / factor - 1
    fac = 2 / (val2 + 1)
    res = mwdx_fast(source, fac)
    return res if sequential else res[-1]

@njit
def mwdx_fast(source, fac):
    if False:
        return 10
    newseries = np.copy(source)
    for i in range(1, source.shape[0]):
        newseries[i] = fac * source[i] + (1 - fac) * newseries[i - 1]
    return newseries