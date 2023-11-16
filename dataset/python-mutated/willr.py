from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def willr(candles: np.ndarray, period: int=14, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        return 10
    "\n    WILLR - Williams' %R\n\n    :param candles: np.ndarray\n    :param period: int - default: 14\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    "
    candles = slice_candles(candles, sequential)
    res = talib.WILLR(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod=period)
    return res if sequential else res[-1]