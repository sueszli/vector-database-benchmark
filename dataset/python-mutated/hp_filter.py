import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from statsmodels.tools.validation import array_like, PandasWrapper

def hpfilter(x, lamb=1600):
    if False:
        i = 10
        return i + 15
    '\n    Hodrick-Prescott filter.\n\n    Parameters\n    ----------\n    x : array_like\n        The time series to filter, 1-d.\n    lamb : float\n        The Hodrick-Prescott smoothing parameter. A value of 1600 is\n        suggested for quarterly data. Ravn and Uhlig suggest using a value\n        of 6.25 (1600/4**4) for annual data and 129600 (1600*3**4) for monthly\n        data.\n\n    Returns\n    -------\n    cycle : ndarray\n        The estimated cycle in the data given lamb.\n    trend : ndarray\n        The estimated trend in the data given lamb.\n\n    See Also\n    --------\n    statsmodels.tsa.filters.bk_filter.bkfilter\n        Baxter-King filter.\n    statsmodels.tsa.filters.cf_filter.cffilter\n        The Christiano Fitzgerald asymmetric, random walk filter.\n    statsmodels.tsa.seasonal.seasonal_decompose\n        Decompose a time series using moving averages.\n    statsmodels.tsa.seasonal.STL\n        Season-Trend decomposition using LOESS.\n\n    Notes\n    -----\n    The HP filter removes a smooth trend, `T`, from the data `x`. by solving\n\n    min sum((x[t] - T[t])**2 + lamb*((T[t+1] - T[t]) - (T[t] - T[t-1]))**2)\n     T   t\n\n    Here we implemented the HP filter as a ridge-regression rule using\n    scipy.sparse. In this sense, the solution can be written as\n\n    T = inv(I + lamb*K\'K)x\n\n    where I is a nobs x nobs identity matrix, and K is a (nobs-2) x nobs matrix\n    such that\n\n    K[i,j] = 1 if i == j or i == j + 2\n    K[i,j] = -2 if i == j + 1\n    K[i,j] = 0 otherwise\n\n    See the notebook `Time Series Filters\n    <../examples/notebooks/generated/tsa_filters.html>`__ for an overview.\n\n    References\n    ----------\n    Hodrick, R.J, and E. C. Prescott. 1980. "Postwar U.S. Business Cycles: An\n        Empirical Investigation." `Carnegie Mellon University discussion\n        paper no. 451`.\n    Ravn, M.O and H. Uhlig. 2002. "Notes On Adjusted the Hodrick-Prescott\n        Filter for the Frequency of Observations." `The Review of Economics and\n        Statistics`, 84(2), 371-80.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> import pandas as pd\n    >>> dta = sm.datasets.macrodata.load_pandas().data\n    >>> index = pd.period_range(\'1959Q1\', \'2009Q3\', freq=\'Q\')\n    >>> dta.set_index(index, inplace=True)\n\n    >>> cycle, trend = sm.tsa.filters.hpfilter(dta.realgdp, 1600)\n    >>> gdp_decomp = dta[[\'realgdp\']]\n    >>> gdp_decomp["cycle"] = cycle\n    >>> gdp_decomp["trend"] = trend\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> gdp_decomp[["realgdp", "trend"]]["2000-03-31":].plot(ax=ax,\n    ...                                                      fontsize=16)\n    >>> plt.show()\n\n    .. plot:: plots/hpf_plot.py\n    '
    pw = PandasWrapper(x)
    x = array_like(x, 'x', ndim=1)
    nobs = len(x)
    I = sparse.eye(nobs, nobs)
    offsets = np.array([0, 1, 2])
    data = np.repeat([[1.0], [-2.0], [1.0]], nobs, axis=1)
    K = sparse.dia_matrix((data, offsets), shape=(nobs - 2, nobs))
    use_umfpack = True
    trend = spsolve(I + lamb * K.T.dot(K), x, use_umfpack=use_umfpack)
    cycle = x - trend
    return (pw.wrap(cycle, append='cycle'), pw.wrap(trend, append='trend'))