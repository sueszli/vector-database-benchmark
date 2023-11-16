from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def medprice(candles: np.ndarray, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    '\n    MEDPRICE - Median Price\n\n    :param candles: np.ndarray\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = talib.MEDPRICE(candles[:, 3], candles[:, 4])
    return res if sequential else res[-1]