import numpy as np
from scipy.signal import fftconvolve
from statsmodels.tools.validation import array_like, PandasWrapper

def bkfilter(x, low=6, high=32, K=12):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filter a time series using the Baxter-King bandpass filter.\n\n    Parameters\n    ----------\n    x : array_like\n        A 1 or 2d ndarray. If 2d, variables are assumed to be in columns.\n    low : float\n        Minimum period for oscillations, ie., Baxter and King suggest that\n        the Burns-Mitchell U.S. business cycle has 6 for quarterly data and\n        1.5 for annual data.\n    high : float\n        Maximum period for oscillations BK suggest that the U.S.\n        business cycle has 32 for quarterly data and 8 for annual data.\n    K : int\n        Lead-lag length of the filter. Baxter and King propose a truncation\n        length of 12 for quarterly data and 3 for annual data.\n\n    Returns\n    -------\n    ndarray\n        The cyclical component of x.\n\n    See Also\n    --------\n    statsmodels.tsa.filters.cf_filter.cffilter\n        The Christiano Fitzgerald asymmetric, random walk filter.\n    statsmodels.tsa.filters.bk_filter.hpfilter\n        Hodrick-Prescott filter.\n    statsmodels.tsa.seasonal.seasonal_decompose\n        Decompose a time series using moving averages.\n    statsmodels.tsa.seasonal.STL\n        Season-Trend decomposition using LOESS.\n\n    Notes\n    -----\n    Returns a centered weighted moving average of the original series. Where\n    the weights a[j] are computed ::\n\n      a[j] = b[j] + theta, for j = 0, +/-1, +/-2, ... +/- K\n      b[0] = (omega_2 - omega_1)/pi\n      b[j] = 1/(pi*j)(sin(omega_2*j)-sin(omega_1*j), for j = +/-1, +/-2,...\n\n    and theta is a normalizing constant ::\n\n      theta = -sum(b)/(2K+1)\n\n    See the notebook `Time Series Filters\n    <../examples/notebooks/generated/tsa_filters.html>`__ for an overview.\n\n    References\n    ----------\n    Baxter, M. and R. G. King. "Measuring Business Cycles: Approximate\n        Band-Pass Filters for Economic Time Series." *Review of Economics and\n        Statistics*, 1999, 81(4), 575-593.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> import pandas as pd\n    >>> dta = sm.datasets.macrodata.load_pandas().data\n    >>> index = pd.DatetimeIndex(start=\'1959Q1\', end=\'2009Q4\', freq=\'Q\')\n    >>> dta.set_index(index, inplace=True)\n\n    >>> cycles = sm.tsa.filters.bkfilter(dta[[\'realinv\']], 6, 24, 12)\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> cycles.plot(ax=ax, style=[\'r--\', \'b-\'])\n    >>> plt.show()\n\n    .. plot:: plots/bkf_plot.py\n    '
    pw = PandasWrapper(x)
    x = array_like(x, 'x', maxdim=2)
    omega_1 = 2.0 * np.pi / high
    omega_2 = 2.0 * np.pi / low
    bweights = np.zeros(2 * K + 1)
    bweights[K] = (omega_2 - omega_1) / np.pi
    j = np.arange(1, int(K) + 1)
    weights = 1 / (np.pi * j) * (np.sin(omega_2 * j) - np.sin(omega_1 * j))
    bweights[K + j] = weights
    bweights[:K] = weights[::-1]
    bweights -= bweights.mean()
    if x.ndim == 2:
        bweights = bweights[:, None]
    x = fftconvolve(x, bweights, mode='valid')
    return pw.wrap(x, append='cycle', trim_start=K, trim_end=K)