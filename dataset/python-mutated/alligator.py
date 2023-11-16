from collections import namedtuple
import numpy as np
from jesse.helpers import get_candle_source, np_shift, slice_candles
AG = namedtuple('AG', ['jaw', 'teeth', 'lips'])

def alligator(candles: np.ndarray, source_type: str='close', sequential: bool=False) -> AG:
    if False:
        return 10
    '\n    Alligator\n\n    :param candles: np.ndarray\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: AG(jaw, teeth, lips)\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    jaw = np_shift(numpy_ewma(source, 13), 8, fill_value=np.nan)
    teeth = np_shift(numpy_ewma(source, 8), 5, fill_value=np.nan)
    lips = np_shift(numpy_ewma(source, 5), 3, fill_value=np.nan)
    if sequential:
        return AG(jaw, teeth, lips)
    else:
        return AG(jaw[-1], teeth[-1], lips[-1])

def numpy_ewma(data: np.ndarray, window: int):
    if False:
        for i in range(10):
            print('nop')
    '\n\n    :param data:\n    :param window:\n    :return:\n    '
    alpha = 1 / window
    n = data.shape[0]
    scale_arr = (1 - alpha) ** (-1 * np.arange(n))
    weights = (1 - alpha) ** np.arange(n)
    pw0 = (1 - alpha) ** (n - 1)
    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    return cumsums * scale_arr[::-1] / weights.cumsum()