from typing import Union
import numpy as np
import talib
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles

def ht_dcperiod(candles: np.ndarray, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        return 10
    '\n    HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period\n\n    :param candles: np.ndarray\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = talib.HT_DCPERIOD(source)
    return res if sequential else res[-1]