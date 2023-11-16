from typing import Union
import numpy as np
import talib
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles

def cfo(candles: np.ndarray, period: int=14, scalar: float=100, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    CFO - Chande Forcast Oscillator\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param scalar: float - default: 100\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = scalar * (source - talib.LINEARREG(source, timeperiod=period))
    res /= source
    if sequential:
        return res
    else:
        return None if np.isnan(res[-1]) else res[-1]