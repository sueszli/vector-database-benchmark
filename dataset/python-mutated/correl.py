from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def correl(candles: np.ndarray, period: int=5, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    "\n    CORREL - Pearson's Correlation Coefficient (r)\n\n    :param candles: np.ndarray\n    :param period: int - default: 5\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    "
    candles = slice_candles(candles, sequential)
    res = talib.CORREL(candles[:, 3], candles[:, 4], timeperiod=period)
    return res if sequential else res[-1]