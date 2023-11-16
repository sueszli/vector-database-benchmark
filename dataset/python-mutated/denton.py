import numpy as np
from numpy import dot, eye, diag_indices, zeros, ones, diag, asarray, r_
from numpy.linalg import solve

def dentonm(indicator, benchmark, freq='aq', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Modified Denton\'s method to convert low-frequency to high-frequency data.\n\n    Uses proportionate first-differences as the penalty function.  See notes.\n\n    Parameters\n    ----------\n    indicator : array_like\n        A low-frequency indicator series.  It is assumed that there are no\n        pre-sample indicators.  Ie., the first indicators line up with\n        the first benchmark.\n    benchmark : array_like\n        The higher frequency benchmark.  A 1d or 2d data series in columns.\n        If 2d, then M series are assumed.\n    freq : str {"aq","qm", "other"}\n        The frequency to use in the conversion.\n\n        * "aq" - Benchmarking an annual series to quarterly.\n        * "mq" - Benchmarking a quarterly series to monthly.\n        * "other" - Custom stride.  A kwarg, k, must be supplied.\n    **kwargs\n        Additional keyword argument. For example:\n\n        * k, an int, the number of high-frequency observations that sum to make\n          an aggregate low-frequency observation. `k` is used with\n          `freq` == "other".\n\n    Returns\n    -------\n    transformed : ndarray\n        The transformed series.\n\n    Examples\n    --------\n    >>> indicator = [50,100,150,100] * 5\n    >>> benchmark = [500,400,300,400,500]\n    >>> benchmarked = dentonm(indicator, benchmark, freq="aq")\n\n    Notes\n    -----\n    Denton\'s method minimizes the distance given by the penalty function, in\n    a least squares sense, between the unknown benchmarked series and the\n    indicator series subject to the condition that the sum of the benchmarked\n    series is equal to the benchmark. The modification allows that the first\n    value not be pre-determined as is the case with Denton\'s original method.\n    If the there is no benchmark provided for the last few indicator\n    observations, then extrapolation is performed using the last\n    benchmark-indicator ratio of the previous period.\n\n    Minimizes sum((X[t]/I[t] - X[t-1]/I[t-1])**2)\n\n    s.t.\n\n    sum(X) = A, for each period.  Where X is the benchmarked series, I is\n    the indicator, and A is the benchmark.\n\n    References\n    ----------\n    Bloem, A.M, Dippelsman, R.J. and Maehle, N.O.  2001 Quarterly National\n        Accounts Manual--Concepts, Data Sources, and Compilation. IMF.\n        http://www.imf.org/external/pubs/ft/qna/2000/Textbook/index.htm\n    Cholette, P. 1988. "Benchmarking systems of socio-economic time series."\n        Statistics Canada, Time Series Research and Analysis Division,\n        Working Paper No TSRA-88-017E.\n    Denton, F.T. 1971. "Adjustment of monthly or quarterly series to annual\n        totals: an approach based on quadratic minimization." Journal of the\n        American Statistical Association. 99-102.\n    '
    indicator = asarray(indicator)
    if indicator.ndim == 1:
        indicator = indicator[:, None]
    benchmark = asarray(benchmark)
    if benchmark.ndim == 1:
        benchmark = benchmark[:, None]
    N = len(indicator)
    m = len(benchmark)
    if freq == 'aq':
        k = 4
    elif freq == 'qm':
        k = 3
    elif freq == 'other':
        k = kwargs.get('k')
        if not k:
            raise ValueError('k must be supplied with freq="other"')
    else:
        raise ValueError('freq %s not understood' % freq)
    n = k * m
    if N > n:
        q = N - n
    else:
        q = 0
    B = np.kron(np.eye(m), ones((k, 1)))
    Zinv = diag(1.0 / indicator.squeeze()[:n])
    HTH = eye(n)
    (diag_idx0, diag_idx1) = diag_indices(n)
    HTH[diag_idx0[1:-1], diag_idx1[1:-1]] += 1
    HTH[diag_idx0[:-1] + 1, diag_idx1[:-1]] = -1
    HTH[diag_idx0[:-1], diag_idx1[:-1] + 1] = -1
    W = dot(dot(Zinv, HTH), Zinv)
    I = zeros((n + m, n + m))
    I[:n, :n] = W
    I[:n, n:] = B
    I[n:, :n] = B.T
    A = zeros((m + n, 1))
    A[-m:] = benchmark
    X = solve(I, A)
    X = X[:-m]
    if q > 0:
        bi = X[n - 1] / indicator[n - 1]
        extrapolated = bi * indicator[n:]
        X = r_[X, extrapolated]
    return X.squeeze()
if __name__ == '__main__':
    indicator = np.array([98.2, 100.8, 102.2, 100.8, 99.0, 101.6, 102.7, 101.5, 100.5, 103.0, 103.5, 101.5])
    benchmark = np.array([4000.0, 4161.4])
    x_imf = dentonm(indicator, benchmark, freq='aq')
    imf_stata = np.array([969.8, 998.4, 1018.3, 1013.4, 1007.2, 1042.9, 1060.3, 1051.0, 1040.6, 1066.5, 1071.7, 1051.0])
    np.testing.assert_almost_equal(imf_stata, x_imf, 1)
    zQ = np.array([50, 100, 150, 100] * 5)
    Y = np.array([500, 400, 300, 400, 500])
    x_denton = dentonm(zQ, Y, freq='aq')
    x_stata = np.array([64.334796, 127.80616, 187.82379, 120.03526, 56.563894, 105.97568, 147.50144, 89.958987, 40.547201, 74.445963, 108.34473, 76.66211, 42.763347, 94.14664, 153.41596, 109.67405, 58.290761, 122.62556, 190.41409, 128.66959])
'\n# Examples from the Denton 1971 paper\nk = 4\nm = 5\nn = m*k\n\nzQ = [50,100,150,100] * m\nY = [500,400,300,400,500]\n\nA = np.eye(n)\nB = block_diag(*(np.ones((k,1)),)*m)\n\nr = Y - B.T.dot(zQ)\n#Ainv = inv(A)\nAinv = A # shortcut for identity\nC = Ainv.dot(B).dot(inv(B.T.dot(Ainv).dot(B)))\nx = zQ + C.dot(r)\n\n# minimize first difference d(x-z)\nR = linalg.tri(n, dtype=float) # R is tril so actually R.T in paper\nAinv = R.dot(R.T)\nC = Ainv.dot(B).dot(inv(B.T.dot(Ainv).dot(B)))\nx1 = zQ + C.dot(r)\n\n# minimize the second difference d**2(x-z)\nAinv = R.dot(Ainv).dot(R.T)\nC = Ainv.dot(B).dot(inv(B.T.dot(Ainv).dot(B)))\nx12 = zQ + C.dot(r)\n\n\n# # do it proportionately (x-z)/z\nZ = np.diag(zQ)\nAinv = np.eye(n)\nC = Z.dot(Ainv).dot(Z).dot(B).dot(inv(B.T.dot(Z).dot(Ainv).dot(Z).dot(B)))\nx11 = zQ + C.dot(r)\n\n# do it proportionately with differencing d((x-z)/z)\nAinv = R.dot(R.T)\nC = Z.dot(Ainv).dot(Z).dot(B).dot(inv(B.T.dot(Z).dot(Ainv).dot(Z).dot(B)))\nx111 = zQ + C.dot(r)\n\nx_stata = np.array([64.334796,127.80616,187.82379,120.03526,56.563894,\n                    105.97568,147.50144,89.958987,40.547201,74.445963,\n                    108.34473,76.66211,42.763347,94.14664,153.41596,\n                    109.67405,58.290761,122.62556,190.41409,128.66959])\n'