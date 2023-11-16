from typing import Union
import numpy as np
import tulipy as ti
from jesse.helpers import get_candle_source, same_length, slice_candles

def fosc(candles: np.ndarray, period: int=5, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    FOSC - Forecast Oscillator\n\n    :param candles: np.ndarray\n    :param period: int - default: 5\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = ti.fosc(np.ascontiguousarray(source), period=period)
    return same_length(candles, res) if sequential else res[-1]