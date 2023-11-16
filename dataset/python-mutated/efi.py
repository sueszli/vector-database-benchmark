from typing import Union
import numpy as np
import talib
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source, same_length, slice_candles

def efi(candles: np.ndarray, period: int=13, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    EFI - Elders Force Index\n\n    :param candles: np.ndarray\n    :param period: int - default: 13\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    dif = efi_fast(source, candles[:, 5])
    res = talib.EMA(dif, timeperiod=period)
    res_with_nan = same_length(candles, res)
    return res_with_nan if sequential else res_with_nan[-1]

@njit
def efi_fast(source, volume):
    if False:
        print('Hello World!')
    dif = np.zeros(source.size - 1)
    for i in range(1, source.size):
        dif[i - 1] = (source[i] - source[i - 1]) * volume[i]
    return dif