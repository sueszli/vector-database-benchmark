import numpy as np
from collections import namedtuple
from jesse.helpers import slice_candles
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
HA = namedtuple('HA', ['open', 'close', 'high', 'low'])

def heikin_ashi_candles(candles: np.ndarray, sequential: bool=False) -> HA:
    if False:
        return 10
    '\n    Heikin Ashi Candles\n    :param candles: np.ndarray\n    :param sequential: bool - default: False\n    :return: float | np.ndarray\n    '
    source = slice_candles(candles, sequential)
    (open, close, high, low) = ha_fast(source[:, [1, 2, 3, 4]])
    if sequential:
        return HA(open, close, high, low)
    else:
        return HA(open[-1], close[-1], high[-1], low[-1])

@njit
def ha_fast(source):
    if False:
        return 10
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    ha_candles = np.full_like(source, np.nan)
    for i in range(1, ha_candles.shape[0]):
        ha_candles[i][OPEN] = (source[i - 1][OPEN] + source[i - 1][CLOSE]) / 2
        ha_candles[i][CLOSE] = (source[i][OPEN] + source[i][CLOSE] + source[i][HIGH] + source[i][LOW]) / 4
        ha_candles[i][HIGH] = max([source[i][HIGH], ha_candles[i][OPEN], ha_candles[i][CLOSE]])
        ha_candles[i][LOW] = min([source[i][LOW], ha_candles[i][OPEN], ha_candles[i][CLOSE]])
    return (ha_candles[:, OPEN], ha_candles[:, CLOSE], ha_candles[:, HIGH], ha_candles[:, LOW])