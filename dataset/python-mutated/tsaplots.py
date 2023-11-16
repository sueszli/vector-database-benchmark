"""Correlation plot functions."""
from statsmodels.compat.pandas import deprecate_kwarg
import calendar
import numpy as np
import pandas as pd
from statsmodels.graphics import utils
from statsmodels.tools.validation import array_like
from statsmodels.tsa.stattools import acf, pacf, ccf

def _prepare_data_corr_plot(x, lags, zero):
    if False:
        for i in range(10):
            print('nop')
    zero = bool(zero)
    irregular = False if zero else True
    if lags is None:
        nobs = x.shape[0]
        lim = min(int(np.ceil(10 * np.log10(nobs))), nobs - 1)
        lags = np.arange(not zero, lim + 1)
    elif np.isscalar(lags):
        lags = np.arange(not zero, int(lags) + 1)
    else:
        irregular = True
        lags = np.asanyarray(lags).astype(int)
    nlags = lags.max(0)
    return (lags, nlags, irregular)

def _plot_corr(ax, title, acf_x, confint, lags, irregular, use_vlines, vlines_kwargs, auto_ylims=False, skip_lag0_confint=True, **kwargs):
    if False:
        i = 10
        return i + 15
    if irregular:
        acf_x = acf_x[lags]
        if confint is not None:
            confint = confint[lags]
    if use_vlines:
        ax.vlines(lags, [0], acf_x, **vlines_kwargs)
        ax.axhline(**kwargs)
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 5)
    if 'ls' not in kwargs:
        kwargs.setdefault('linestyle', 'None')
    ax.margins(0.05)
    ax.plot(lags, acf_x, **kwargs)
    ax.set_title(title)
    ax.set_ylim(-1, 1)
    if auto_ylims:
        ax.set_ylim(1.25 * np.minimum(min(acf_x), min(confint[:, 0] - acf_x)), 1.25 * np.maximum(max(acf_x), max(confint[:, 1] - acf_x)))
    if confint is not None:
        if skip_lag0_confint and lags[0] == 0:
            lags = lags[1:]
            confint = confint[1:]
            acf_x = acf_x[1:]
        lags = lags.astype(float)
        lags[np.argmin(lags)] -= 0.5
        lags[np.argmax(lags)] += 0.5
        ax.fill_between(lags, confint[:, 0] - acf_x, confint[:, 1] - acf_x, alpha=0.25)

@deprecate_kwarg('unbiased', 'adjusted')
def plot_acf(x, ax=None, lags=None, *, alpha=0.05, use_vlines=True, adjusted=False, fft=False, missing='none', title='Autocorrelation', zero=True, auto_ylims=False, bartlett_confint=True, vlines_kwargs=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Plot the autocorrelation function\n\n    Plots lags on the horizontal and the correlations on vertical axis.\n\n    Parameters\n    ----------\n    x : array_like\n        Array of time-series values\n    ax : AxesSubplot, optional\n        If given, this subplot is used to plot in instead of a new figure being\n        created.\n    lags : {int, array_like}, optional\n        An int or array of lag values, used on horizontal axis. Uses\n        np.arange(lags) when lags is an int.  If not provided,\n        ``lags=np.arange(len(corr))`` is used.\n    alpha : scalar, optional\n        If a number is given, the confidence intervals for the given level are\n        returned. For instance if alpha=.05, 95 % confidence intervals are\n        returned where the standard deviation is computed according to\n        Bartlett\'s formula. If None, no confidence intervals are plotted.\n    use_vlines : bool, optional\n        If True, vertical lines and markers are plotted.\n        If False, only markers are plotted.  The default marker is \'o\'; it can\n        be overridden with a ``marker`` kwarg.\n    adjusted : bool\n        If True, then denominators for autocovariance are n-k, otherwise n\n    fft : bool, optional\n        If True, computes the ACF via FFT.\n    missing : str, optional\n        A string in [\'none\', \'raise\', \'conservative\', \'drop\'] specifying how\n        the NaNs are to be treated.\n    title : str, optional\n        Title to place on plot.  Default is \'Autocorrelation\'\n    zero : bool, optional\n        Flag indicating whether to include the 0-lag autocorrelation.\n        Default is True.\n    auto_ylims : bool, optional\n        If True, adjusts automatically the y-axis limits to ACF values.\n    bartlett_confint : bool, default True\n        Confidence intervals for ACF values are generally placed at 2\n        standard errors around r_k. The formula used for standard error\n        depends upon the situation. If the autocorrelations are being used\n        to test for randomness of residuals as part of the ARIMA routine,\n        the standard errors are determined assuming the residuals are white\n        noise. The approximate formula for any lag is that standard error\n        of each r_k = 1/sqrt(N). See section 9.4 of [1] for more details on\n        the 1/sqrt(N) result. For more elementary discussion, see section\n        5.3.2 in [2].\n        For the ACF of raw data, the standard error at a lag k is\n        found as if the right model was an MA(k-1). This allows the\n        possible interpretation that if all autocorrelations past a\n        certain lag are within the limits, the model might be an MA of\n        order defined by the last significant autocorrelation. In this\n        case, a moving average model is assumed for the data and the\n        standard errors for the confidence intervals should be\n        generated using Bartlett\'s formula. For more details on\n        Bartlett formula result, see section 7.2 in [1].\n    vlines_kwargs : dict, optional\n        Optional dictionary of keyword arguments that are passed to vlines.\n    **kwargs : kwargs, optional\n        Optional keyword arguments that are directly passed on to the\n        Matplotlib ``plot`` and ``axhline`` functions.\n\n    Returns\n    -------\n    Figure\n        If `ax` is None, the created figure.  Otherwise the figure to which\n        `ax` is connected.\n\n    See Also\n    --------\n    matplotlib.pyplot.xcorr\n    matplotlib.pyplot.acorr\n\n    Notes\n    -----\n    Adapted from matplotlib\'s `xcorr`.\n\n    Data are plotted as ``plot(lags, corr, **kwargs)``\n\n    kwargs is used to pass matplotlib optional arguments to both the line\n    tracing the autocorrelations and for the horizontal line at 0. These\n    options must be valid for a Line2D object.\n\n    vlines_kwargs is used to pass additional optional arguments to the\n    vertical lines connecting each autocorrelation to the axis.  These options\n    must be valid for a LineCollection object.\n\n    References\n    ----------\n    [1] Brockwell and Davis, 1987. Time Series Theory and Methods\n    [2] Brockwell and Davis, 2010. Introduction to Time Series and\n    Forecasting, 2nd edition.\n\n    Examples\n    --------\n    >>> import pandas as pd\n    >>> import matplotlib.pyplot as plt\n    >>> import statsmodels.api as sm\n\n    >>> dta = sm.datasets.sunspots.load_pandas().data\n    >>> dta.index = pd.Index(sm.tsa.datetools.dates_from_range(\'1700\', \'2008\'))\n    >>> del dta["YEAR"]\n    >>> sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)\n    >>> plt.show()\n\n    .. plot:: plots/graphics_tsa_plot_acf.py\n    '
    (fig, ax) = utils.create_mpl_ax(ax)
    (lags, nlags, irregular) = _prepare_data_corr_plot(x, lags, zero)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs
    confint = None
    acf_x = acf(x, nlags=nlags, alpha=alpha, fft=fft, bartlett_confint=bartlett_confint, adjusted=adjusted, missing=missing)
    if alpha is not None:
        (acf_x, confint) = acf_x[:2]
    _plot_corr(ax, title, acf_x, confint, lags, irregular, use_vlines, vlines_kwargs, auto_ylims=auto_ylims, **kwargs)
    return fig

def plot_pacf(x, ax=None, lags=None, alpha=0.05, method='ywm', use_vlines=True, title='Partial Autocorrelation', zero=True, vlines_kwargs=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Plot the partial autocorrelation function\n\n    Parameters\n    ----------\n    x : array_like\n        Array of time-series values\n    ax : AxesSubplot, optional\n        If given, this subplot is used to plot in instead of a new figure being\n        created.\n    lags : {int, array_like}, optional\n        An int or array of lag values, used on horizontal axis. Uses\n        np.arange(lags) when lags is an int.  If not provided,\n        ``lags=np.arange(len(corr))`` is used.\n    alpha : float, optional\n        If a number is given, the confidence intervals for the given level are\n        returned. For instance if alpha=.05, 95 % confidence intervals are\n        returned where the standard deviation is computed according to\n        1/sqrt(len(x))\n    method : str\n        Specifies which method for the calculations to use:\n\n        - "ywm" or "ywmle" : Yule-Walker without adjustment. Default.\n        - "yw" or "ywadjusted" : Yule-Walker with sample-size adjustment in\n          denominator for acovf. Default.\n        - "ols" : regression of time series on lags of it and on constant.\n        - "ols-inefficient" : regression of time series on lags using a single\n          common sample to estimate all pacf coefficients.\n        - "ols-adjusted" : regression of time series on lags with a bias\n          adjustment.\n        - "ld" or "ldadjusted" : Levinson-Durbin recursion with bias\n          correction.\n        - "ldb" or "ldbiased" : Levinson-Durbin recursion without bias\n          correction.\n\n    use_vlines : bool, optional\n        If True, vertical lines and markers are plotted.\n        If False, only markers are plotted.  The default marker is \'o\'; it can\n        be overridden with a ``marker`` kwarg.\n    title : str, optional\n        Title to place on plot.  Default is \'Partial Autocorrelation\'\n    zero : bool, optional\n        Flag indicating whether to include the 0-lag autocorrelation.\n        Default is True.\n    vlines_kwargs : dict, optional\n        Optional dictionary of keyword arguments that are passed to vlines.\n    **kwargs : kwargs, optional\n        Optional keyword arguments that are directly passed on to the\n        Matplotlib ``plot`` and ``axhline`` functions.\n\n    Returns\n    -------\n    Figure\n        If `ax` is None, the created figure.  Otherwise the figure to which\n        `ax` is connected.\n\n    See Also\n    --------\n    matplotlib.pyplot.xcorr\n    matplotlib.pyplot.acorr\n\n    Notes\n    -----\n    Plots lags on the horizontal and the correlations on vertical axis.\n    Adapted from matplotlib\'s `xcorr`.\n\n    Data are plotted as ``plot(lags, corr, **kwargs)``\n\n    kwargs is used to pass matplotlib optional arguments to both the line\n    tracing the autocorrelations and for the horizontal line at 0. These\n    options must be valid for a Line2D object.\n\n    vlines_kwargs is used to pass additional optional arguments to the\n    vertical lines connecting each autocorrelation to the axis.  These options\n    must be valid for a LineCollection object.\n\n    Examples\n    --------\n    >>> import pandas as pd\n    >>> import matplotlib.pyplot as plt\n    >>> import statsmodels.api as sm\n\n    >>> dta = sm.datasets.sunspots.load_pandas().data\n    >>> dta.index = pd.Index(sm.tsa.datetools.dates_from_range(\'1700\', \'2008\'))\n    >>> del dta["YEAR"]\n    >>> sm.graphics.tsa.plot_pacf(dta.values.squeeze(), lags=40, method="ywm")\n    >>> plt.show()\n\n    .. plot:: plots/graphics_tsa_plot_pacf.py\n    '
    (fig, ax) = utils.create_mpl_ax(ax)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs
    (lags, nlags, irregular) = _prepare_data_corr_plot(x, lags, zero)
    confint = None
    if alpha is None:
        acf_x = pacf(x, nlags=nlags, alpha=alpha, method=method)
    else:
        (acf_x, confint) = pacf(x, nlags=nlags, alpha=alpha, method=method)
    _plot_corr(ax, title, acf_x, confint, lags, irregular, use_vlines, vlines_kwargs, **kwargs)
    return fig

def plot_ccf(x, y, *, ax=None, lags=None, negative_lags=False, alpha=0.05, use_vlines=True, adjusted=False, fft=False, title='Cross-correlation', auto_ylims=False, vlines_kwargs=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Plot the cross-correlation function\n\n    Correlations between ``x`` and the lags of ``y`` are calculated.\n\n    The lags are shown on the horizontal axis and the correlations\n    on the vertical axis.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Arrays of time-series values.\n    ax : AxesSubplot, optional\n        If given, this subplot is used to plot in, otherwise a new figure with\n        one subplot is created.\n    lags : {int, array_like}, optional\n        An int or array of lag values, used on the horizontal axis. Uses\n        ``np.arange(lags)`` when lags is an int.  If not provided,\n        ``lags=np.arange(len(corr))`` is used.\n    negative_lags: bool, optional\n        If True, negative lags are shown on the horizontal axis.\n    alpha : scalar, optional\n        If a number is given, the confidence intervals for the given level are\n        plotted, e.g. if alpha=.05, 95 % confidence intervals are shown.\n        If None, confidence intervals are not shown on the plot.\n    use_vlines : bool, optional\n        If True, shows vertical lines and markers for the correlation values.\n        If False, only shows markers.  The default marker is \'o\'; it can\n        be overridden with a ``marker`` kwarg.\n    adjusted : bool\n        If True, then denominators for cross-correlations are n-k, otherwise n.\n    fft : bool, optional\n        If True, computes the CCF via FFT.\n    title : str, optional\n        Title to place on plot. Default is \'Cross-correlation\'.\n    auto_ylims : bool, optional\n        If True, adjusts automatically the vertical axis limits to CCF values.\n    vlines_kwargs : dict, optional\n        Optional dictionary of keyword arguments that are passed to vlines.\n    **kwargs : kwargs, optional\n        Optional keyword arguments that are directly passed on to the\n        Matplotlib ``plot`` and ``axhline`` functions.\n\n    Returns\n    -------\n    Figure\n        The figure where the plot is drawn. This is either an existing figure\n        if the `ax` argument is provided, or a newly created figure\n        if `ax` is None.\n\n    See Also\n    --------\n    See notes and references for statsmodels.graphics.tsaplots.plot_acf\n\n    Examples\n    --------\n    >>> import pandas as pd\n    >>> import matplotlib.pyplot as plt\n    >>> import statsmodels.api as sm\n\n    >>> dta = sm.datasets.macrodata.load_pandas().data\n    >>> diffed = dta.diff().dropna()\n    >>> sm.graphics.tsa.plot_ccf(diffed["unemp"], diffed["infl"])\n    >>> plt.show()\n    '
    (fig, ax) = utils.create_mpl_ax(ax)
    (lags, nlags, irregular) = _prepare_data_corr_plot(x, lags, True)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs
    if negative_lags:
        lags = -lags
    ccf_res = ccf(x, y, adjusted=adjusted, fft=fft, alpha=alpha, nlags=nlags + 1)
    if alpha is not None:
        (ccf_xy, confint) = ccf_res
    else:
        ccf_xy = ccf_res
        confint = None
    _plot_corr(ax, title, ccf_xy, confint, lags, irregular, use_vlines, vlines_kwargs, auto_ylims=auto_ylims, skip_lag0_confint=False, **kwargs)
    return fig

def plot_accf_grid(x, *, varnames=None, fig=None, lags=None, negative_lags=True, alpha=0.05, use_vlines=True, adjusted=False, fft=False, missing='none', zero=True, auto_ylims=False, bartlett_confint=False, vlines_kwargs=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Plot auto/cross-correlation grid\n\n    Plots lags on the horizontal axis and the correlations\n    on the vertical axis of each graph.\n\n    Parameters\n    ----------\n    x : array_like\n        2D array of time-series values: rows are observations,\n        columns are variables.\n    varnames: sequence of str, optional\n        Variable names to use in plot titles. If ``x`` is a pandas dataframe\n        and ``varnames`` is provided, it overrides the column names\n        of the dataframe. If ``varnames`` is not provided and ``x`` is not\n        a dataframe, variable names ``x[0]``, ``x[1]``, etc. are generated.\n    fig : Matplotlib figure instance, optional\n        If given, this figure is used to plot in, otherwise a new figure\n        is created.\n    lags : {int, array_like}, optional\n        An int or array of lag values, used on horizontal axes. Uses\n        ``np.arange(lags)`` when lags is an int.  If not provided,\n        ``lags=np.arange(len(corr))`` is used.\n    negative_lags: bool, optional\n        If True, negative lags are shown on the horizontal axes of plots\n        below the main diagonal.\n    alpha : scalar, optional\n        If a number is given, the confidence intervals for the given level are\n        plotted, e.g. if alpha=.05, 95 % confidence intervals are shown.\n        If None, confidence intervals are not shown on the plot.\n    use_vlines : bool, optional\n        If True, shows vertical lines and markers for the correlation values.\n        If False, only shows markers.  The default marker is \'o\'; it can\n        be overridden with a ``marker`` kwarg.\n    adjusted : bool\n        If True, then denominators for correlations are n-k, otherwise n.\n    fft : bool, optional\n        If True, computes the ACF via FFT.\n    missing : str, optional\n        A string in [\'none\', \'raise\', \'conservative\', \'drop\'] specifying how\n        NaNs are to be treated.\n    zero : bool, optional\n        Flag indicating whether to include the 0-lag autocorrelations\n        (which are always equal to 1). Default is True.\n    auto_ylims : bool, optional\n        If True, adjusts automatically the vertical axis limits\n        to correlation values.\n    bartlett_confint : bool, default False\n        If True, use Bartlett\'s formula to calculate confidence intervals\n        in auto-correlation plots. See the description of ``plot_acf`` for\n        details. This argument does not affect cross-correlation plots.\n    vlines_kwargs : dict, optional\n        Optional dictionary of keyword arguments that are passed to vlines.\n    **kwargs : kwargs, optional\n        Optional keyword arguments that are directly passed on to the\n        Matplotlib ``plot`` and ``axhline`` functions.\n\n    Returns\n    -------\n    Figure\n        If `fig` is None, the created figure.  Otherwise, `fig` is returned.\n        Plots on the grid show the cross-correlation of the row variable\n        with the lags of the column variable.\n\n    See Also\n    --------\n    See notes and references for statsmodels.graphics.tsaplots\n\n    Examples\n    --------\n    >>> import pandas as pd\n    >>> import matplotlib.pyplot as plt\n    >>> import statsmodels.api as sm\n\n    >>> dta = sm.datasets.macrodata.load_pandas().data\n    >>> diffed = dta.diff().dropna()\n    >>> sm.graphics.tsa.plot_accf_grid(diffed[["unemp", "infl"]])\n    >>> plt.show()\n    '
    from statsmodels.tools.data import _is_using_pandas
    array_like(x, 'x', ndim=2)
    m = x.shape[1]
    fig = utils.create_mpl_fig(fig)
    gs = fig.add_gridspec(m, m)
    if _is_using_pandas(x, None):
        varnames = varnames or list(x.columns)

        def get_var(i):
            if False:
                i = 10
                return i + 15
            return x.iloc[:, i]
    else:
        varnames = varnames or [f'x[{i}]' for i in range(m)]
        x = np.asarray(x)

        def get_var(i):
            if False:
                while True:
                    i = 10
            return x[:, i]
    for i in range(m):
        for j in range(m):
            ax = fig.add_subplot(gs[i, j])
            if i == j:
                plot_acf(get_var(i), ax=ax, title=f'ACF({varnames[i]})', lags=lags, alpha=alpha, use_vlines=use_vlines, adjusted=adjusted, fft=fft, missing=missing, zero=zero, auto_ylims=auto_ylims, bartlett_confint=bartlett_confint, vlines_kwargs=vlines_kwargs, **kwargs)
            else:
                plot_ccf(get_var(i), get_var(j), ax=ax, title=f'CCF({varnames[i]}, {varnames[j]})', lags=lags, negative_lags=negative_lags and i > j, alpha=alpha, use_vlines=use_vlines, adjusted=adjusted, fft=fft, auto_ylims=auto_ylims, vlines_kwargs=vlines_kwargs, **kwargs)
    return fig

def seasonal_plot(grouped_x, xticklabels, ylabel=None, ax=None):
    if False:
        return 10
    '\n    Consider using one of month_plot or quarter_plot unless you need\n    irregular plotting.\n\n    Parameters\n    ----------\n    grouped_x : iterable of DataFrames\n        Should be a GroupBy object (or similar pair of group_names and groups\n        as DataFrames) with a DatetimeIndex or PeriodIndex\n    xticklabels : list of str\n        List of season labels, one for each group.\n    ylabel : str\n        Lable for y axis\n    ax : AxesSubplot, optional\n        If given, this subplot is used to plot in instead of a new figure being\n        created.\n    '
    (fig, ax) = utils.create_mpl_ax(ax)
    start = 0
    ticks = []
    for (season, df) in grouped_x:
        df = df.copy()
        df.sort_index()
        nobs = len(df)
        x_plot = np.arange(start, start + nobs)
        ticks.append(x_plot.mean())
        ax.plot(x_plot, df.values, 'k')
        ax.hlines(df.values.mean(), x_plot[0], x_plot[-1], colors='r', linewidth=3)
        start += nobs
    ax.set_xticks(ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.margins(0.1, 0.05)
    return fig

def month_plot(x, dates=None, ylabel=None, ax=None):
    if False:
        while True:
            i = 10
    "\n    Seasonal plot of monthly data.\n\n    Parameters\n    ----------\n    x : array_like\n        Seasonal data to plot. If dates is None, x must be a pandas object\n        with a PeriodIndex or DatetimeIndex with a monthly frequency.\n    dates : array_like, optional\n        If `x` is not a pandas object, then dates must be supplied.\n    ylabel : str, optional\n        The label for the y-axis. Will attempt to use the `name` attribute\n        of the Series.\n    ax : Axes, optional\n        Existing axes instance.\n\n    Returns\n    -------\n    Figure\n       If `ax` is provided, the Figure instance attached to `ax`. Otherwise\n       a new Figure instance.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> import pandas as pd\n\n    >>> dta = sm.datasets.elnino.load_pandas().data\n    >>> dta['YEAR'] = dta.YEAR.astype(int).astype(str)\n    >>> dta = dta.set_index('YEAR').T.unstack()\n    >>> dates = pd.to_datetime(list(map(lambda x: '-'.join(x) + '-1',\n    ...                                 dta.index.values)))\n    >>> dta.index = pd.DatetimeIndex(dates, freq='MS')\n    >>> fig = sm.graphics.tsa.month_plot(dta)\n\n    .. plot:: plots/graphics_tsa_month_plot.py\n    "
    if dates is None:
        from statsmodels.tools.data import _check_period_index
        _check_period_index(x, freq='M')
    else:
        x = pd.Series(x, index=pd.PeriodIndex(dates, freq='M'))
    xticklabels = list(calendar.month_abbr)[1:]
    return seasonal_plot(x.groupby(lambda y: y.month), xticklabels, ylabel=ylabel, ax=ax)

def quarter_plot(x, dates=None, ylabel=None, ax=None):
    if False:
        while True:
            i = 10
    "\n    Seasonal plot of quarterly data\n\n    Parameters\n    ----------\n    x : array_like\n        Seasonal data to plot. If dates is None, x must be a pandas object\n        with a PeriodIndex or DatetimeIndex with a monthly frequency.\n    dates : array_like, optional\n        If `x` is not a pandas object, then dates must be supplied.\n    ylabel : str, optional\n        The label for the y-axis. Will attempt to use the `name` attribute\n        of the Series.\n    ax : matplotlib.axes, optional\n        Existing axes instance.\n\n    Returns\n    -------\n    Figure\n       If `ax` is provided, the Figure instance attached to `ax`. Otherwise\n       a new Figure instance.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> import pandas as pd\n\n    >>> dta = sm.datasets.elnino.load_pandas().data\n    >>> dta['YEAR'] = dta.YEAR.astype(int).astype(str)\n    >>> dta = dta.set_index('YEAR').T.unstack()\n    >>> dates = pd.to_datetime(list(map(lambda x: '-'.join(x) + '-1',\n    ...                                 dta.index.values)))\n    >>> dta.index = dates.to_period('Q')\n    >>> fig = sm.graphics.tsa.quarter_plot(dta)\n\n    .. plot:: plots/graphics_tsa_quarter_plot.py\n    "
    if dates is None:
        from statsmodels.tools.data import _check_period_index
        _check_period_index(x, freq='Q')
    else:
        x = pd.Series(x, index=pd.PeriodIndex(dates, freq='Q'))
    xticklabels = ['q1', 'q2', 'q3', 'q4']
    return seasonal_plot(x.groupby(lambda y: y.quarter), xticklabels, ylabel=ylabel, ax=ax)

def plot_predict(result, start=None, end=None, dynamic=False, alpha=0.05, ax=None, **predict_kwargs):
    if False:
        print('Hello World!')
    '\n\n    Parameters\n    ----------\n    result : Result\n        Any model result supporting ``get_prediction``.\n    start : int, str, or datetime, optional\n        Zero-indexed observation number at which to start forecasting,\n        i.e., the first forecast is start. Can also be a date string to\n        parse or a datetime type. Default is the the zeroth observation.\n    end : int, str, or datetime, optional\n        Zero-indexed observation number at which to end forecasting, i.e.,\n        the last forecast is end. Can also be a date string to\n        parse or a datetime type. However, if the dates index does not\n        have a fixed frequency, end must be an integer index if you\n        want out of sample prediction. Default is the last observation in\n        the sample.\n    dynamic : bool, int, str, or datetime, optional\n        Integer offset relative to `start` at which to begin dynamic\n        prediction. Can also be an absolute date string to parse or a\n        datetime type (these are not interpreted as offsets).\n        Prior to this observation, true endogenous values will be used for\n        prediction; starting with this observation and continuing through\n        the end of prediction, forecasted endogenous values will be used\n        instead.\n    alpha : {float, None}\n        The tail probability not covered by the confidence interval. Must\n        be in (0, 1). Confidence interval is constructed assuming normally\n        distributed shocks. If None, figure will not show the confidence\n        interval.\n    ax : AxesSubplot\n        matplotlib Axes instance to use\n    **predict_kwargs\n        Any additional keyword arguments to pass to ``result.get_prediction``.\n\n    Returns\n    -------\n    Figure\n        matplotlib Figure containing the prediction plot\n    '
    from statsmodels.graphics.utils import _import_mpl, create_mpl_ax
    _ = _import_mpl()
    (fig, ax) = create_mpl_ax(ax)
    from statsmodels.tsa.base.prediction import PredictionResults
    pred: PredictionResults = result.get_prediction(start=start, end=end, dynamic=dynamic, **predict_kwargs)
    mean = pred.predicted_mean
    if isinstance(mean, (pd.Series, pd.DataFrame)):
        x = mean.index
        mean.plot(ax=ax, label='forecast')
    else:
        x = np.arange(mean.shape[0])
        ax.plot(x, mean)
    if alpha is not None:
        label = f'{1 - alpha:.0%} confidence interval'
        ci = pred.conf_int(alpha)
        conf_int = np.asarray(ci)
        ax.fill_between(x, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.5, label=label)
    ax.legend(loc='best')
    return fig