from typing import Union
import numpy as np
import talib
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from jesse.helpers import slice_candles

def chande(candles: np.ndarray, period: int=22, mult: float=3.0, direction: str='long', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    Chandelier Exits\n\n    :param candles: np.ndarray\n    :param period: int - default: 22\n    :param mult: float - default: 3.0\n    :param direction: str - default: "long"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    candles_close = candles[:, 2]
    candles_high = candles[:, 3]
    candles_low = candles[:, 4]
    atr = talib.ATR(candles_high, candles_low, candles_close, timeperiod=period)
    if direction == 'long':
        maxp = filter1d_same(candles_high, period, 'max')
        result = maxp - atr * mult
    elif direction == 'short':
        maxp = filter1d_same(candles_low, period, 'min')
        result = maxp + atr * mult
    else:
        print("The last parameter must be 'short' or 'long'")
    return result if sequential else result[-1]

def filter1d_same(a: np.ndarray, W: int, max_or_min: str, fillna=np.nan):
    if False:
        return 10
    out_dtype = np.full(0, fillna).dtype
    hW = (W - 1) // 2
    if max_or_min == 'max':
        out = maximum_filter1d(a, size=W, origin=hW)
    else:
        out = minimum_filter1d(a, size=W, origin=hW)
    if out.dtype is out_dtype:
        out[:W - 1] = fillna
    else:
        out = np.concatenate((np.full(W - 1, fillna), out[W - 1:]))
    return out