"""
This module contains an implementation of an exponentially weighted moving average based on sample size.
The inspiration and context for this code was from a blog post by writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""
import numpy as np
from numba import jit
from numba import float64
from numba import int64

@jit((float64[:], int64), nopython=False, nogil=True)
def ewma(arr_in, window):
    if False:
        for i in range(10):
            print('nop')
    "\n    Exponentially weighted moving average specified by a decay ``window`` to provide better adjustments for\n    small windows via:\n        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /\n               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).\n\n    :param arr_in: (np.ndarray), (float64) A single dimensional numpy array\n    :param window: (int64) The decay window, or 'span'\n    :return: (np.ndarray) The EWMA vector, same length / shape as ``arr_in``\n    "
    pass