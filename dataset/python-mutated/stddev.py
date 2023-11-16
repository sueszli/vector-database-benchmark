from typing import Union
import numpy as np
import talib
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles

def stddev(candles: np.ndarray, period: int=5, nbdev: float=1, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    '\n    STDDEV - Standard Deviation\n\n    :param candles: np.ndarray\n    :param period: int - default: 5\n    :param nbdev: float - default: 1\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = talib.STDDEV(source, timeperiod=period, nbdev=nbdev)
    return res if sequential else res[-1]