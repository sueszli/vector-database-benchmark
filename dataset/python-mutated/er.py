from typing import Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from jesse.helpers import get_candle_source, slice_candles, same_length

def er(candles: np.ndarray, period: int=5, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    '\n    ER - The Kaufman Efficiency indicator\n\n    :param candles: np.ndarray\n    :param period: int - default: 5\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    change = np.abs(np.diff(source, period))
    abs_dif = np.abs(np.diff(source))
    swv = sliding_window_view(abs_dif, window_shape=period)
    volatility = swv.sum()
    res = change / volatility
    return same_length(candles, res) if sequential else res[-1]