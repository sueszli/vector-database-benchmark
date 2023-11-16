from typing import Union
import numpy as np
import tulipy as ti
from jesse.helpers import get_candle_source, same_length
from jesse.helpers import slice_candles

def pvi(candles: np.ndarray, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        return 10
    '\n    PVI - Positive Volume Index\n\n    :param candles: np.ndarray\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = ti.pvi(np.ascontiguousarray(source), np.ascontiguousarray(candles[:, 5]))
    return same_length(candles, res) if sequential else res[-1]