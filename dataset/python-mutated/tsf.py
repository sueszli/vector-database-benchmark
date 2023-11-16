from typing import Union
import numpy as np
import talib
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles

def tsf(candles: np.ndarray, period: int=14, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        while True:
            i = 10
    '\n    TSF - Time Series Forecast\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = talib.TSF(source, timeperiod=period)
    return res if sequential else res[-1]