"""
Created on Thu Oct 21 21:45:24 2010

Author: josef-pktd
"""
import numpy as np
from scipy import signal

def armaloop(arcoefs, macoefs, x):
    if False:
        print('Hello World!')
    'get arma recursion in simple loop\n\n    for simplicity assumes that ma polynomial is not longer than the ar-polynomial\n\n    Parameters\n    ----------\n    arcoefs : array_like\n        autoregressive coefficients in right hand side parameterization\n    macoefs : array_like\n        moving average coefficients, without leading 1\n\n    Returns\n    -------\n    y : ndarray\n        predicted values, initial values are the same as the observed values\n    e : ndarray\n        predicted residuals, zero for initial observations\n\n    Notes\n    -----\n    Except for the treatment of initial observations this is the same as using\n    scipy.signal.lfilter, which is much faster. Written for testing only\n    '
    arcoefs_r = np.asarray(arcoefs)
    macoefs_r = np.asarray(macoefs)
    x = np.asarray(x)
    nobs = x.shape[0]
    arlag = arcoefs_r.shape[0]
    malag = macoefs_r.shape[0]
    maxlag = max(arlag, malag)
    print(maxlag)
    y = np.zeros(x.shape, float)
    e = np.zeros(x.shape, float)
    y[:maxlag] = x[:maxlag]
    for t in range(arlag, maxlag):
        y[t] = (x[t - arlag:t] * arcoefs_r).sum(0) + (e[:t] * macoefs_r[:t]).sum(0)
        e[t] = x[t] - y[t]
    for t in range(maxlag, nobs):
        y[t] = (x[t - arlag:t] * arcoefs_r).sum(0) + (e[t - malag:t] * macoefs_r).sum(0)
        e[t] = x[t] - y[t]
    return (y, e)
(arcoefs, macoefs) = (-np.array([1, -0.8, 0.2])[1:], np.array([1.0, 0.5, 0.1])[1:])
print(armaloop(arcoefs, macoefs, np.ones(10)))
print(armaloop([0.8], [], np.ones(10)))
print(armaloop([0.8], [], np.arange(2, 10)))
(y, e) = armaloop([0.1], [0.8], np.arange(2, 10))
print(e)
print(signal.lfilter(np.array([1, -0.1]), np.array([1.0, 0.8]), np.arange(2, 10)))
(y, e) = armaloop([], [0.8], np.ones(10))
print(e)
print(signal.lfilter(np.array([1, -0.0]), np.array([1.0, 0.8]), np.ones(10)))
ic = signal.lfiltic(np.array([1, -0.1]), np.array([1.0, 0.8]), np.ones([0]), np.array([1]))
print(signal.lfilter(np.array([1, -0.1]), np.array([1.0, 0.8]), np.ones(10), zi=ic))
zi = signal.lfilter_zi(np.array([1, -0.8, 0.2]), np.array([1.0, 0, 0]))
print(signal.lfilter(np.array([1, -0.1]), np.array([1.0, 0.8]), np.ones(10), zi=zi))
print(signal.filtfilt(np.array([1, -0.8]), np.array([1.0]), np.ones(10)))