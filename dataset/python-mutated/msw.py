from collections import namedtuple
import numpy as np
import tulipy as ti
from jesse.helpers import get_candle_source, same_length, slice_candles
MSW = namedtuple('MSW', ['sine', 'lead'])

def msw(candles: np.ndarray, period: int=5, source_type: str='close', sequential: bool=False) -> MSW:
    if False:
        return 10
    '\n    MSW - Mesa Sine Wave\n\n    :param candles: np.ndarray\n    :param period: int - default: 5\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: MSW(sine, lead)\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    (msw_sine, msw_lead) = ti.msw(np.ascontiguousarray(source), period=period)
    s = same_length(candles, msw_sine)
    l = same_length(candles, msw_lead)
    if sequential:
        return MSW(s, l)
    else:
        return MSW(s[-1], l[-1])