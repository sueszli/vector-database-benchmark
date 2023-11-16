from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def obv(candles: np.ndarray, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    '\n    OBV - On Balance Volume\n\n    :param candles: np.ndarray\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = talib.OBV(candles[:, 2], candles[:, 5])
    return res if sequential else res[-1]