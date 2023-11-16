"""
Seasonal Decomposition by Moving Averages
"""
import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tools.validation import PandasWrapper, array_like
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.stl.mstl import MSTL
from statsmodels.tsa.tsatools import freq_to_period
__all__ = ['STL', 'seasonal_decompose', 'seasonal_mean', 'DecomposeResult', 'MSTL']

def _extrapolate_trend(trend, npoints):
    if False:
        i = 10
        return i + 15
    "\n    Replace nan values on trend's end-points with least-squares extrapolated\n    values with regression considering npoints closest defined points.\n    "
    front = next((i for (i, vals) in enumerate(trend) if not np.any(np.isnan(vals))))
    back = trend.shape[0] - 1 - next((i for (i, vals) in enumerate(trend[::-1]) if not np.any(np.isnan(vals))))
    front_last = min(front + npoints, back)
    back_first = max(front, back - npoints)
    (k, n) = np.linalg.lstsq(np.c_[np.arange(front, front_last), np.ones(front_last - front)], trend[front:front_last], rcond=-1)[0]
    extra = (np.arange(0, front) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[:front] = extra
    (k, n) = np.linalg.lstsq(np.c_[np.arange(back_first, back), np.ones(back - back_first)], trend[back_first:back], rcond=-1)[0]
    extra = (np.arange(back + 1, trend.shape[0]) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[back + 1:] = extra
    return trend

def seasonal_mean(x, period):
    if False:
        return 10
    '\n    Return means for each period in x. period is an int that gives the\n    number of periods per cycle. E.g., 12 for monthly. NaNs are ignored\n    in the mean.\n    '
    return np.array([pd_nanmean(x[i::period], axis=0) for i in range(period)])

def seasonal_decompose(x, model='additive', filt=None, period=None, two_sided=True, extrapolate_trend=0):
    if False:
        return 10
    '\n    Seasonal decomposition using moving averages.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series. If 2d, individual series are in columns. x must contain 2\n        complete cycles.\n    model : {"additive", "multiplicative"}, optional\n        Type of seasonal component. Abbreviations are accepted.\n    filt : array_like, optional\n        The filter coefficients for filtering out the seasonal component.\n        The concrete moving average method used in filtering is determined by\n        two_sided.\n    period : int, optional\n        Period of the series. Must be used if x is not a pandas object or if\n        the index of x does not have  a frequency. Overrides default\n        periodicity of x if x is a pandas object with a timeseries index.\n    two_sided : bool, optional\n        The moving average method used in filtering.\n        If True (default), a centered moving average is computed using the\n        filt. If False, the filter coefficients are for past values only.\n    extrapolate_trend : int or \'freq\', optional\n        If set to > 0, the trend resulting from the convolution is\n        linear least-squares extrapolated on both ends (or the single one\n        if two_sided is False) considering this many (+1) closest points.\n        If set to \'freq\', use `freq` closest points. Setting this parameter\n        results in no NaN values in trend or resid components.\n\n    Returns\n    -------\n    DecomposeResult\n        A object with seasonal, trend, and resid attributes.\n\n    See Also\n    --------\n    statsmodels.tsa.filters.bk_filter.bkfilter\n        Baxter-King filter.\n    statsmodels.tsa.filters.cf_filter.cffilter\n        Christiano-Fitzgerald asymmetric, random walk filter.\n    statsmodels.tsa.filters.hp_filter.hpfilter\n        Hodrick-Prescott filter.\n    statsmodels.tsa.filters.convolution_filter\n        Linear filtering via convolution.\n    statsmodels.tsa.seasonal.STL\n        Season-Trend decomposition using LOESS.\n\n    Notes\n    -----\n    This is a naive decomposition. More sophisticated methods should\n    be preferred.\n\n    The additive model is Y[t] = T[t] + S[t] + e[t]\n\n    The multiplicative model is Y[t] = T[t] * S[t] * e[t]\n\n    The results are obtained by first estimating the trend by applying\n    a convolution filter to the data. The trend is then removed from the\n    series and the average of this de-trended series for each period is\n    the returned seasonal component.\n    '
    pfreq = period
    pw = PandasWrapper(x)
    if period is None:
        pfreq = getattr(getattr(x, 'index', None), 'inferred_freq', None)
    x = array_like(x, 'x', maxdim=2)
    nobs = len(x)
    if not np.all(np.isfinite(x)):
        raise ValueError('This function does not handle missing values')
    if model.startswith('m'):
        if np.any(x <= 0):
            raise ValueError('Multiplicative seasonality is not appropriate for zero and negative values')
    if period is None:
        if pfreq is not None:
            pfreq = freq_to_period(pfreq)
            period = pfreq
        else:
            raise ValueError('You must specify a period or x must be a pandas object with a PeriodIndex or a DatetimeIndex with a freq not set to None')
    if x.shape[0] < 2 * pfreq:
        raise ValueError(f'x must have 2 complete cycles requires {2 * pfreq} observations. x only has {x.shape[0]} observation(s)')
    if filt is None:
        if period % 2 == 0:
            filt = np.array([0.5] + [1] * (period - 1) + [0.5]) / period
        else:
            filt = np.repeat(1.0 / period, period)
    nsides = int(two_sided) + 1
    trend = convolution_filter(x, filt, nsides)
    if extrapolate_trend == 'freq':
        extrapolate_trend = period - 1
    if extrapolate_trend > 0:
        trend = _extrapolate_trend(trend, extrapolate_trend + 1)
    if model.startswith('m'):
        detrended = x / trend
    else:
        detrended = x - trend
    period_averages = seasonal_mean(detrended, period)
    if model.startswith('m'):
        period_averages /= np.mean(period_averages, axis=0)
    else:
        period_averages -= np.mean(period_averages, axis=0)
    seasonal = np.tile(period_averages.T, nobs // period + 1).T[:nobs]
    if model.startswith('m'):
        resid = x / seasonal / trend
    else:
        resid = detrended - seasonal
    results = []
    for (s, name) in zip((seasonal, trend, resid, x), ('seasonal', 'trend', 'resid', None)):
        results.append(pw.wrap(s.squeeze(), columns=name))
    return DecomposeResult(seasonal=results[0], trend=results[1], resid=results[2], observed=results[3])

class DecomposeResult:
    """
    Results class for seasonal decompositions

    Parameters
    ----------
    observed : array_like
        The data series that has been decomposed.
    seasonal : array_like
        The seasonal component of the data series.
    trend : array_like
        The trend component of the data series.
    resid : array_like
        The residual component of the data series.
    weights : array_like, optional
        The weights used to reduce outlier influence.
    """

    def __init__(self, observed, seasonal, trend, resid, weights=None):
        if False:
            while True:
                i = 10
        self._seasonal = seasonal
        self._trend = trend
        if weights is None:
            weights = np.ones_like(observed)
            if isinstance(observed, pd.Series):
                weights = pd.Series(weights, index=observed.index, name='weights')
        self._weights = weights
        self._resid = resid
        self._observed = observed

    @property
    def observed(self):
        if False:
            return 10
        'Observed data'
        return self._observed

    @property
    def seasonal(self):
        if False:
            while True:
                i = 10
        'The estimated seasonal component'
        return self._seasonal

    @property
    def trend(self):
        if False:
            i = 10
            return i + 15
        'The estimated trend component'
        return self._trend

    @property
    def resid(self):
        if False:
            for i in range(10):
                print('nop')
        'The estimated residuals'
        return self._resid

    @property
    def weights(self):
        if False:
            print('Hello World!')
        'The weights used in the robust estimation'
        return self._weights

    @property
    def nobs(self):
        if False:
            print('Hello World!')
        'Number of observations'
        return self._observed.shape

    def plot(self, observed=True, seasonal=True, trend=True, resid=True, weights=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Plot estimated components\n\n        Parameters\n        ----------\n        observed : bool\n            Include the observed series in the plot\n        seasonal : bool\n            Include the seasonal component in the plot\n        trend : bool\n            Include the trend component in the plot\n        resid : bool\n            Include the residual in the plot\n        weights : bool\n            Include the weights in the plot (if any)\n\n        Returns\n        -------\n        matplotlib.figure.Figure\n            The figure instance that containing the plot.\n        '
        from pandas.plotting import register_matplotlib_converters
        from statsmodels.graphics.utils import _import_mpl
        plt = _import_mpl()
        register_matplotlib_converters()
        series = [(self._observed, 'Observed')] if observed else []
        series += [(self.trend, 'trend')] if trend else []
        if self.seasonal.ndim == 1:
            series += [(self.seasonal, 'seasonal')] if seasonal else []
        elif self.seasonal.ndim > 1:
            if isinstance(self.seasonal, pd.DataFrame):
                for col in self.seasonal.columns:
                    series += [(self.seasonal[col], 'seasonal')] if seasonal else []
            else:
                for i in range(self.seasonal.shape[1]):
                    series += [(self.seasonal[:, i], 'seasonal')] if seasonal else []
        series += [(self.resid, 'residual')] if resid else []
        series += [(self.weights, 'weights')] if weights else []
        if isinstance(self._observed, (pd.DataFrame, pd.Series)):
            nobs = self._observed.shape[0]
            xlim = (self._observed.index[0], self._observed.index[nobs - 1])
        else:
            xlim = (0, self._observed.shape[0] - 1)
        (fig, axs) = plt.subplots(len(series), 1, sharex=True)
        for (i, (ax, (series, def_name))) in enumerate(zip(axs, series)):
            if def_name != 'residual':
                ax.plot(series)
            else:
                ax.plot(series, marker='o', linestyle='none')
                ax.plot(xlim, (0, 0), color='#000000', zorder=-3)
            name = getattr(series, 'name', def_name)
            if def_name != 'Observed':
                name = name.capitalize()
            title = ax.set_title if i == 0 and observed else ax.set_ylabel
            title(name)
            ax.set_xlim(xlim)
        fig.tight_layout()
        return fig