from typing import Union
import numpy as np
import tulipy as ti
from jesse.helpers import get_candle_source, same_length
from jesse.helpers import slice_candles

def vidya(candles: np.ndarray, short_period: int=2, long_period: int=5, alpha: float=0.2, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    '\n    VIDYA - Variable Index Dynamic Average\n\n    :param candles: np.ndarray\n    :param short_period: int - default: 2\n    :param long_period: int - default: 5\n    :param alpha: float - default: 0.2\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    res = ti.vidya(np.ascontiguousarray(source), short_period=short_period, long_period=long_period, alpha=alpha)
    return same_length(candles, res) if sequential else res[-1]