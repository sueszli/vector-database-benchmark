from collections import namedtuple
import numpy as np
import tulipy as ti
from jesse.helpers import slice_candles, same_length
FisherTransform = namedtuple('FisherTransform', ['fisher', 'signal'])

def fisher(candles: np.ndarray, period: int=9, sequential: bool=False) -> FisherTransform:
    if False:
        while True:
            i = 10
    '\n    The Fisher Transform helps identify price reversals.\n\n    :param candles: np.ndarray\n    :param period: int - default: 9\n    :param sequential: bool - default: False\n\n    :return: FisherTransform(fisher, signal)\n    '
    candles = slice_candles(candles, sequential)
    (fisher_val, fisher_signal) = ti.fisher(np.ascontiguousarray(candles[:, 3]), np.ascontiguousarray(candles[:, 4]), period=period)
    if sequential:
        return FisherTransform(same_length(candles, fisher_val), same_length(candles, fisher_signal))
    else:
        return FisherTransform(fisher_val[-1], fisher_signal[-1])