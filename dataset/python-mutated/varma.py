"""VAR and VARMA process

this does not actually do much, trying out a version for a time loop

alternative representation:
* textbook, different blocks in matrices
* Kalman filter
* VAR, VARX and ARX could be calculated with signal.lfilter
  only tried some examples, not implemented

TODO: try minimizing sum of squares of (Y-Yhat)

Note: filter has smallest lag at end of array and largest lag at beginning,
    be careful for asymmetric lags coefficients
    check this again if it is consistently used


changes
2009-09-08 : separated from movstat.py

Author : josefpkt
License : BSD
"""
import numpy as np
from scipy import signal

def VAR(x, B, const=0):
    if False:
        return 10
    ' multivariate linear filter\n\n    Parameters\n    ----------\n    x: (TxK) array\n        columns are variables, rows are observations for time period\n    B: (PxKxK) array\n        b_t-1 is bottom "row", b_t-P is top "row" when printing\n        B(:,:,0) is lag polynomial matrix for variable 1\n        B(:,:,k) is lag polynomial matrix for variable k\n        B(p,:,k) is pth lag for variable k\n        B[p,:,:].T corresponds to A_p in Wikipedia\n    const : float or array (not tested)\n        constant added to autoregression\n\n    Returns\n    -------\n    xhat: (TxK) array\n        filtered, predicted values of x array\n\n    Notes\n    -----\n    xhat(t,i) = sum{_p}sum{_k} { x(t-P:t,:) .* B(:,:,i) }  for all i = 0,K-1, for all t=p..T\n\n    xhat does not include the forecasting observation, xhat(T+1),\n    xhat is 1 row shorter than signal.correlate\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Vector_Autoregression\n    https://en.wikipedia.org/wiki/General_matrix_notation_of_a_VAR(p)\n    '
    p = B.shape[0]
    T = x.shape[0]
    xhat = np.zeros(x.shape)
    for t in range(p, T):
        xhat[t, :] = const + (x[t - p:t, :, np.newaxis] * B).sum(axis=1).sum(axis=0)
    return xhat

def VARMA(x, B, C, const=0):
    if False:
        return 10
    ' multivariate linear filter\n\n    x (TxK)\n    B (PxKxK)\n\n    xhat(t,i) = sum{_p}sum{_k} { x(t-P:t,:) .* B(:,:,i) } +\n                sum{_q}sum{_k} { e(t-Q:t,:) .* C(:,:,i) }for all i = 0,K-1\n\n    '
    P = B.shape[0]
    Q = C.shape[0]
    T = x.shape[0]
    xhat = np.zeros(x.shape)
    e = np.zeros(x.shape)
    start = max(P, Q)
    for t in range(start, T):
        xhat[t, :] = const + (x[t - P:t, :, np.newaxis] * B).sum(axis=1).sum(axis=0) + (e[t - Q:t, :, np.newaxis] * C).sum(axis=1).sum(axis=0)
        e[t, :] = x[t, :] - xhat[t, :]
    return (xhat, e)
if __name__ == '__main__':
    T = 20
    K = 2
    P = 3
    x = np.column_stack([np.arange(T)] * K)
    B = np.ones((P, K, K))
    B[:, :, 1] = [[0, 0], [0, 0], [0, 1]]
    xhat = VAR(x, B)
    print(np.all(xhat[P:, 0] == np.correlate(x[:-1, 0], np.ones(P)) * 2))
    T = 20
    K = 2
    Q = 2
    P = 3
    const = 1
    x = np.column_stack([np.arange(T)] * K)
    B = np.ones((P, K, K))
    B[:, :, 1] = [[0, 0], [0, 0], [0, 1]]
    C = np.zeros((Q, K, K))
    xhat1 = VAR(x, B, const=const)
    (xhat2, err2) = VARMA(x, B, C, const=const)
    print(np.all(xhat2 == xhat1))
    print(np.all(xhat2[P:, 0] == np.correlate(x[:-1, 0], np.ones(P)) * 2 + const))
    C[1, 1, 1] = 0.5
    (xhat3, err3) = VARMA(x, B, C)
    x = np.r_[np.zeros((P, K)), x]
    (xhat4, err4) = VARMA(x, B, C)
    C[1, 1, 1] = 1
    B[:, :, 1] = [[0, 0], [0, 0], [0, 1]]
    (xhat5, err5) = VARMA(x, B, C)
    x0 = np.column_stack([np.arange(T), 2 * np.arange(T)])
    B[:, :, 0] = np.ones((P, K))
    B[:, :, 1] = np.ones((P, K))
    B[1, 1, 1] = 0
    xhat0 = VAR(x0, B)
    xcorr00 = signal.correlate(x0, B[:, :, 0])
    xcorr01 = signal.correlate(x0, B[:, :, 1])
    print(np.all(signal.correlate(x0, B[:, :, 0], 'valid')[:-1, 0] == xhat0[P:, 0]))
    print(np.all(signal.correlate(x0, B[:, :, 1], 'valid')[:-1, 0] == xhat0[P:, 1]))
    from statsmodels.tsa.stattools import acovf, acf
    aav = acovf(x[:, 0])
    print(aav[0] == np.var(x[:, 0]))
    aac = acf(x[:, 0])