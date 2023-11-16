from collections import namedtuple
import numpy as np
import talib
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import slice_candles
SuperTrend = namedtuple('SuperTrend', ['trend', 'changed'])

def supertrend(candles: np.ndarray, period: int=10, factor: float=3, sequential: bool=False) -> SuperTrend:
    if False:
        i = 10
        return i + 15
    '\n    SuperTrend\n    :param candles: np.ndarray\n    :param period: int - default=14\n    :param factor: float - default=3\n    :param sequential: bool - default=False\n    :return: SuperTrend(trend, changed)\n    '
    candles = slice_candles(candles, sequential)
    atr = talib.ATR(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod=period)
    (super_trend, changed) = supertrend_fast(candles, atr, factor, period)
    if sequential:
        return SuperTrend(super_trend, changed)
    else:
        return SuperTrend(super_trend[-1], changed[-1])

@njit
def supertrend_fast(candles, atr, factor, period):
    if False:
        i = 10
        return i + 15
    upper_basic = (candles[:, 3] + candles[:, 4]) / 2 + factor * atr
    lower_basic = (candles[:, 3] + candles[:, 4]) / 2 - factor * atr
    upper_band = upper_basic
    lower_band = lower_basic
    super_trend = np.zeros(len(candles))
    changed = np.zeros(len(candles))
    for i in range(period, len(candles)):
        prevClose = candles[:, 2][i - 1]
        prevUpperBand = upper_band[i - 1]
        currUpperBasic = upper_basic[i]
        if prevClose <= prevUpperBand:
            upper_band[i] = min(currUpperBasic, prevUpperBand)
        prevLowerBand = lower_band[i - 1]
        currLowerBasic = lower_basic[i]
        if prevClose >= prevLowerBand:
            lower_band[i] = max(currLowerBasic, prevLowerBand)
        if prevClose <= prevUpperBand:
            super_trend[i - 1] = prevUpperBand
        else:
            super_trend[i - 1] = prevLowerBand
        prevSuperTrend = super_trend[i - 1]
    for i in range(period, len(candles)):
        prevClose = candles[:, 2][i - 1]
        prevUpperBand = upper_band[i - 1]
        currUpperBand = upper_band[i]
        prevLowerBand = lower_band[i - 1]
        currLowerBand = lower_band[i]
        prevSuperTrend = super_trend[i - 1]
        if prevSuperTrend == prevUpperBand:
            if candles[:, 2][i] <= currUpperBand:
                super_trend[i] = currUpperBand
                changed[i] = False
            else:
                super_trend[i] = currLowerBand
                changed[i] = True
        elif prevSuperTrend == prevLowerBand:
            if candles[:, 2][i] >= currLowerBand:
                super_trend[i] = currLowerBand
                changed[i] = False
            else:
                super_trend[i] = currUpperBand
                changed[i] = True
    return (super_trend, changed)