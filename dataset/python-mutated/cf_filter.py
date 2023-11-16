import numpy as np
from statsmodels.tools.validation import PandasWrapper, array_like

def cffilter(x, low=6, high=32, drift=True):
    if False:
        return 10
    '\n    Christiano Fitzgerald asymmetric, random walk filter.\n\n    Parameters\n    ----------\n    x : array_like\n        The 1 or 2d array to filter. If 2d, variables are assumed to be in\n        columns.\n    low : float\n        Minimum period of oscillations. Features below low periodicity are\n        filtered out. Default is 6 for quarterly data, giving a 1.5 year\n        periodicity.\n    high : float\n        Maximum period of oscillations. Features above high periodicity are\n        filtered out. Default is 32 for quarterly data, giving an 8 year\n        periodicity.\n    drift : bool\n        Whether or not to remove a trend from the data. The trend is estimated\n        as np.arange(nobs)*(x[-1] - x[0])/(len(x)-1).\n\n    Returns\n    -------\n    cycle : array_like\n        The features of x between the periodicities low and high.\n    trend : array_like\n        The trend in the data with the cycles removed.\n\n    See Also\n    --------\n    statsmodels.tsa.filters.bk_filter.bkfilter\n        Baxter-King filter.\n    statsmodels.tsa.filters.bk_filter.hpfilter\n        Hodrick-Prescott filter.\n    statsmodels.tsa.seasonal.seasonal_decompose\n        Decompose a time series using moving averages.\n    statsmodels.tsa.seasonal.STL\n        Season-Trend decomposition using LOESS.\n\n    Notes\n    -----\n    See the notebook `Time Series Filters\n    <../examples/notebooks/generated/tsa_filters.html>`__ for an overview.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> import pandas as pd\n    >>> dta = sm.datasets.macrodata.load_pandas().data\n    >>> index = pd.DatetimeIndex(start=\'1959Q1\', end=\'2009Q4\', freq=\'Q\')\n    >>> dta.set_index(index, inplace=True)\n\n    >>> cf_cycles, cf_trend = sm.tsa.filters.cffilter(dta[["infl", "unemp"]])\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> cf_cycles.plot(ax=ax, style=[\'r--\', \'b-\'])\n    >>> plt.show()\n\n    .. plot:: plots/cff_plot.py\n    '
    if low < 2:
        raise ValueError('low must be >= 2')
    pw = PandasWrapper(x)
    x = array_like(x, 'x', ndim=2)
    (nobs, nseries) = x.shape
    a = 2 * np.pi / high
    b = 2 * np.pi / low
    if drift:
        x = x - np.arange(nobs)[:, None] * (x[-1] - x[0]) / (nobs - 1)
    J = np.arange(1, nobs + 1)
    Bj = (np.sin(b * J) - np.sin(a * J)) / (np.pi * J)
    B0 = (b - a) / np.pi
    Bj = np.r_[B0, Bj][:, None]
    y = np.zeros((nobs, nseries))
    for i in range(nobs):
        B = -0.5 * Bj[0] - np.sum(Bj[1:-i - 2])
        A = -Bj[0] - np.sum(Bj[1:-i - 2]) - np.sum(Bj[1:i]) - B
        y[i] = Bj[0] * x[i] + np.dot(Bj[1:-i - 2].T, x[i + 1:-1]) + B * x[-1] + np.dot(Bj[1:i].T, x[1:i][::-1]) + A * x[0]
    y = y.squeeze()
    (cycle, trend) = (y.squeeze(), x.squeeze() - y)
    return (pw.wrap(cycle, append='cycle'), pw.wrap(trend, append='trend'))
if __name__ == '__main__':
    import statsmodels as sm
    dta = sm.datasets.macrodata.load().data[['infl', 'tbilrate']].view((float, 2))[1:]
    (cycle, trend) = cffilter(dta, 6, 32, drift=True)
    dta = sm.datasets.macrodata.load().data['tbilrate'][1:]
    (cycle2, trend2) = cffilter(dta, 6, 32, drift=True)