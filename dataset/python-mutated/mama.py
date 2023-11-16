from collections import namedtuple
import numpy as np
import talib
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles
MAMA = namedtuple('MAMA', ['mama', 'fama'])

def mama(candles: np.ndarray, fastlimit: float=0.5, slowlimit: float=0.05, source_type: str='close', sequential: bool=False) -> MAMA:
    if False:
        i = 10
        return i + 15
    '\n    MAMA - MESA Adaptive Moving Average\n\n    :param candles: np.ndarray\n    :param fastlimit: float - default: 0.5\n    :param slowlimit: float - default: 0.05\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: MAMA(mama, fama)\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    (mama_val, fama) = talib.MAMA(source, fastlimit=fastlimit, slowlimit=slowlimit)
    if sequential:
        return MAMA(mama_val, fama)
    else:
        return MAMA(mama_val[-1], fama[-1])