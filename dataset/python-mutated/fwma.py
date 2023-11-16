from math import fabs
from typing import Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from jesse.helpers import get_candle_source, slice_candles, same_length

def fwma(candles: np.ndarray, period: int=5, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Fibonacci\'s Weighted Moving Average (FWMA)\n\n    :param candles: np.ndarray\n    :param period: int - default: 5\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)
    fibs = fibonacci(n=period)
    swv = sliding_window_view(source, window_shape=period)
    res = np.average(swv, weights=fibs, axis=-1)
    return same_length(candles, res) if sequential else res[-1]

def fibonacci(n: int=2) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Fibonacci Sequence as a numpy array'
    n = int(fabs(n)) if n >= 0 else 2
    n -= 1
    (a, b) = (1, 1)
    result = np.array([a])
    for _ in range(n):
        (a, b) = (b, a + b)
        result = np.append(result, a)
    fib_sum = np.sum(result)
    if fib_sum > 0:
        return result / fib_sum
    else:
        return result