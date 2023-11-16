from collections import namedtuple
import numpy as np
import talib
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles
DamianiVolatmeter = namedtuple('DamianiVolatmeter', ['vol', 'anti'])

def damiani_volatmeter(candles: np.ndarray, vis_atr: int=13, vis_std: int=20, sed_atr: int=40, sed_std: int=100, threshold: float=1.4, source_type: str='close', sequential: bool=False) -> DamianiVolatmeter:
    if False:
        print('Hello World!')
    '\n    Damiani Volatmeter\n\n    :param candles: np.ndarray\n    :param vis_atr: int - default: 13\n    :param vis_std: int - default: 20\n    :param sed_atr: int - default: 40\n    :param sed_std: int - default: 100\n    :param threshold: float - default: 1.4\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    atrvis = talib.ATR(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod=vis_atr)
    atrsed = talib.ATR(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod=sed_atr)
    (vol, t) = damiani_volatmeter_fast(source, sed_std, atrvis, atrsed, vis_std, threshold)
    if sequential:
        return DamianiVolatmeter(vol, t)
    else:
        return DamianiVolatmeter(vol[-1], t[-1])

@njit
def damiani_volatmeter_fast(source, sed_std, atrvis, atrsed, vis_std, threshold):
    if False:
        while True:
            i = 10
    lag_s = 0.5
    vol = np.full_like(source, 0)
    t = np.full_like(source, 0)
    for i in range(source.shape[0]):
        if i >= sed_std:
            vol[i] = atrvis[i] / atrsed[i] + lag_s * (vol[i - 1] - vol[i - 3])
            anti_thres = np.std(source[i - vis_std:i]) / np.std(source[i - sed_std:i])
            t[i] = threshold - anti_thres
    return (vol, t)