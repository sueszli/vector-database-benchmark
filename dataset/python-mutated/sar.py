from typing import Union
import numpy as np
import talib
from jesse.helpers import slice_candles

def sar(candles: np.ndarray, acceleration: float=0.02, maximum: float=0.2, sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        return 10
    '\n    SAR - Parabolic SAR\n\n    :param candles: np.ndarray\n    :param acceleration: float - default: 0.02\n    :param maximum: float - default: 0.2\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    res = talib.SAR(candles[:, 3], candles[:, 4], acceleration=acceleration, maximum=maximum)
    return res if sequential else res[-1]