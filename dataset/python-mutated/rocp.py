from typing import Union
import numpy as np
import talib
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles

def rocp(candles: np.ndarray, period: int=10, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice\n\n    :param candles: np.ndarray\n    :param period: int - default: 10\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = talib.ROCP(source, timeperiod=period)
    return res if sequential else res[-1]