from collections import namedtuple
import numpy as np
import talib as ta
from jesse.helpers import get_candle_source, slice_candles
Wavetrend = namedtuple('Wavetrend', ['wt1', 'wt2', 'wtCrossUp', 'wtCrossDown', 'wtOversold', 'wtOverbought', 'wtVwap'])

def wt(candles: np.ndarray, wtchannellen: int=9, wtaveragelen: int=12, wtmalen: int=3, oblevel: int=53, oslevel: int=-53, source_type: str='hlc3', sequential: bool=False) -> Wavetrend:
    if False:
        i = 10
        return i + 15
    '\n    Wavetrend indicator\n\n    :param candles: np.ndarray\n    :param wtchannellen:  int - default: 9\n    :param wtaveragelen: int - default: 12\n    :param wtmalen: int - default: 3\n    :param oblevel: int - default: 53\n    :param oslevel: int - default: -53\n    :param source_type: str - default: "hlc3"\n    :param sequential: bool - default: False\n\n    :return: Wavetrend\n    '
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type=source_type)
    esa = ta.EMA(src, wtchannellen)
    de = ta.EMA(abs(src - esa), wtchannellen)
    ci = (src - esa) / (0.015 * de)
    wt1 = ta.EMA(ci, wtaveragelen)
    wt2 = ta.SMA(wt1, wtmalen)
    wtVwap = wt1 - wt2
    wtOversold = wt2 <= oslevel
    wtOverbought = wt2 >= oblevel
    wtCrossUp = wt2 - wt1 <= 0
    wtCrossDown = wt2 - wt1 >= 0
    if sequential:
        return Wavetrend(wt1, wt2, wtCrossUp, wtCrossDown, wtOversold, wtOverbought, wtVwap)
    else:
        return Wavetrend(wt1[-1], wt2[-1], wtCrossUp[-1], wtCrossDown[-1], wtOversold[-1], wtOverbought[-1], wtVwap[-1])