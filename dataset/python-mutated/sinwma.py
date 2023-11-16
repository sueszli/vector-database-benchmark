from typing import Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from jesse.helpers import get_candle_source, slice_candles, same_length

def sinwma(candles: np.ndarray, period: int=14, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    Sine Weighted Moving Average (SINWMA)\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    sines = np.array([np.sin((i + 1) * np.pi / (period + 1)) for i in range(period)])
    w = sines / sines.sum()
    swv = sliding_window_view(source, window_shape=period)
    res = np.average(swv, weights=w, axis=-1)
    return same_length(candles, res) if sequential else res[-1]