from typing import Union
import numpy as np
from jesse.helpers import get_candle_source, np_shift, slice_candles

def vpt(candles: np.ndarray, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    Volume Price Trend (VPT)\n\n    :param candles: np.ndarray\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    vpt_val = candles[:, 5] * ((source - np_shift(source, 1, fill_value=np.nan)) / np_shift(source, 1, fill_value=np.nan))
    res = np_shift(vpt_val, 1, fill_value=np.nan) + vpt_val
    return res if sequential else res[-1]