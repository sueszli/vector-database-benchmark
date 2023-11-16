from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval
if TYPE_CHECKING:
    from typing import Literal
    import numpy.typing as npt
__all__ = ['ecdf', 'logrank']

@dataclass
class EmpiricalDistributionFunction:
    """An empirical distribution function produced by `scipy.stats.ecdf`

    Attributes
    ----------
    quantiles : ndarray
        The unique values of the sample from which the
        `EmpiricalDistributionFunction` was estimated.
    probabilities : ndarray
        The point estimates of the cumulative distribution function (CDF) or
        its complement, the survival function (SF), corresponding with
        `quantiles`.
    """
    quantiles: np.ndarray
    probabilities: np.ndarray
    _n: np.ndarray = field(repr=False)
    _d: np.ndarray = field(repr=False)
    _sf: np.ndarray = field(repr=False)
    _kind: str = field(repr=False)

    def __init__(self, q, p, n, d, kind):
        if False:
            for i in range(10):
                print('nop')
        self.probabilities = p
        self.quantiles = q
        self._n = n
        self._d = d
        self._sf = p if kind == 'sf' else 1 - p
        self._kind = kind
        f0 = 1 if kind == 'sf' else 0
        f1 = 1 - f0
        x = np.insert(q, [0, len(q)], [-np.inf, np.inf])
        y = np.insert(p, [0, len(p)], [f0, f1])
        self._f = interpolate.interp1d(x, y, kind='previous', assume_sorted=True)

    def evaluate(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate the empirical CDF/SF function at the input.\n\n        Parameters\n        ----------\n        x : ndarray\n            Argument to the CDF/SF\n\n        Returns\n        -------\n        y : ndarray\n            The CDF/SF evaluated at the input\n        '
        return self._f(x)

    def plot(self, ax=None, **matplotlib_kwargs):
        if False:
            return 10
        "Plot the empirical distribution function\n\n        Available only if ``matplotlib`` is installed.\n\n        Parameters\n        ----------\n        ax : matplotlib.axes.Axes\n            Axes object to draw the plot onto, otherwise uses the current Axes.\n\n        **matplotlib_kwargs : dict, optional\n            Keyword arguments passed directly to `matplotlib.axes.Axes.step`.\n            Unless overridden, ``where='post'``.\n\n        Returns\n        -------\n        lines : list of `matplotlib.lines.Line2D`\n            Objects representing the plotted data\n        "
        try:
            import matplotlib
        except ModuleNotFoundError as exc:
            message = 'matplotlib must be installed to use method `plot`.'
            raise ModuleNotFoundError(message) from exc
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        kwargs = {'where': 'post'}
        kwargs.update(matplotlib_kwargs)
        delta = np.ptp(self.quantiles) * 0.05
        q = self.quantiles
        q = [q[0] - delta] + list(q) + [q[-1] + delta]
        return ax.step(q, self.evaluate(q), **kwargs)

    def confidence_interval(self, confidence_level=0.95, *, method='linear'):
        if False:
            i = 10
            return i + 15
        'Compute a confidence interval around the CDF/SF point estimate\n\n        Parameters\n        ----------\n        confidence_level : float, default: 0.95\n            Confidence level for the computed confidence interval\n\n        method : str, {"linear", "log-log"}\n            Method used to compute the confidence interval. Options are\n            "linear" for the conventional Greenwood confidence interval\n            (default)  and "log-log" for the "exponential Greenwood",\n            log-negative-log-transformed confidence interval.\n\n        Returns\n        -------\n        ci : ``ConfidenceInterval``\n            An object with attributes ``low`` and ``high``, instances of\n            `~scipy.stats._result_classes.EmpiricalDistributionFunction` that\n            represent the lower and upper bounds (respectively) of the\n            confidence interval.\n\n        Notes\n        -----\n        Confidence intervals are computed according to the Greenwood formula\n        (``method=\'linear\'``) or the more recent "exponential Greenwood"\n        formula (``method=\'log-log\'``) as described in [1]_. The conventional\n        Greenwood formula can result in lower confidence limits less than 0\n        and upper confidence limits greater than 1; these are clipped to the\n        unit interval. NaNs may be produced by either method; these are\n        features of the formulas.\n\n        References\n        ----------\n        .. [1] Sawyer, Stanley. "The Greenwood and Exponential Greenwood\n               Confidence Intervals in Survival Analysis."\n               https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf\n\n        '
        message = 'Confidence interval bounds do not implement a `confidence_interval` method.'
        if self._n is None:
            raise NotImplementedError(message)
        methods = {'linear': self._linear_ci, 'log-log': self._loglog_ci}
        message = f'`method` must be one of {set(methods)}.'
        if method.lower() not in methods:
            raise ValueError(message)
        message = '`confidence_level` must be a scalar between 0 and 1.'
        confidence_level = np.asarray(confidence_level)[()]
        if confidence_level.shape or not 0 <= confidence_level <= 1:
            raise ValueError(message)
        method_fun = methods[method.lower()]
        (low, high) = method_fun(confidence_level)
        message = 'The confidence interval is undefined at some observations. This is a feature of the mathematical formula used, not an error in its implementation.'
        if np.any(np.isnan(low) | np.isnan(high)):
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        (low, high) = (np.clip(low, 0, 1), np.clip(high, 0, 1))
        low = EmpiricalDistributionFunction(self.quantiles, low, None, None, self._kind)
        high = EmpiricalDistributionFunction(self.quantiles, high, None, None, self._kind)
        return ConfidenceInterval(low, high)

    def _linear_ci(self, confidence_level):
        if False:
            print('Hello World!')
        (sf, d, n) = (self._sf, self._d, self._n)
        with np.errstate(divide='ignore', invalid='ignore'):
            var = sf ** 2 * np.cumsum(d / (n * (n - d)))
        se = np.sqrt(var)
        z = special.ndtri(1 / 2 + confidence_level / 2)
        z_se = z * se
        low = self.probabilities - z_se
        high = self.probabilities + z_se
        return (low, high)

    def _loglog_ci(self, confidence_level):
        if False:
            print('Hello World!')
        (sf, d, n) = (self._sf, self._d, self._n)
        with np.errstate(divide='ignore', invalid='ignore'):
            var = 1 / np.log(sf) ** 2 * np.cumsum(d / (n * (n - d)))
        se = np.sqrt(var)
        z = special.ndtri(1 / 2 + confidence_level / 2)
        with np.errstate(divide='ignore'):
            lnl_points = np.log(-np.log(sf))
        z_se = z * se
        low = np.exp(-np.exp(lnl_points + z_se))
        high = np.exp(-np.exp(lnl_points - z_se))
        if self._kind == 'cdf':
            (low, high) = (1 - high, 1 - low)
        return (low, high)

@dataclass
class ECDFResult:
    """ Result object returned by `scipy.stats.ecdf`

    Attributes
    ----------
    cdf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
        An object representing the empirical cumulative distribution function.
    sf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
        An object representing the complement of the empirical cumulative
        distribution function.
    """
    cdf: EmpiricalDistributionFunction
    sf: EmpiricalDistributionFunction

    def __init__(self, q, cdf, sf, n, d):
        if False:
            print('Hello World!')
        self.cdf = EmpiricalDistributionFunction(q, cdf, n, d, 'cdf')
        self.sf = EmpiricalDistributionFunction(q, sf, n, d, 'sf')

def _iv_CensoredData(sample: npt.ArrayLike | CensoredData, param_name: str='sample') -> CensoredData:
    if False:
        print('Hello World!')
    'Attempt to convert `sample` to `CensoredData`.'
    if not isinstance(sample, CensoredData):
        try:
            sample = CensoredData(uncensored=sample)
        except ValueError as e:
            message = str(e).replace('uncensored', param_name)
            raise type(e)(message) from e
    return sample

def ecdf(sample: npt.ArrayLike | CensoredData) -> ECDFResult:
    if False:
        while True:
            i = 10
    'Empirical cumulative distribution function of a sample.\n\n    The empirical cumulative distribution function (ECDF) is a step function\n    estimate of the CDF of the distribution underlying a sample. This function\n    returns objects representing both the empirical distribution function and\n    its complement, the empirical survival function.\n\n    Parameters\n    ----------\n    sample : 1D array_like or `scipy.stats.CensoredData`\n        Besides array_like, instances of `scipy.stats.CensoredData` containing\n        uncensored and right-censored observations are supported. Currently,\n        other instances of `scipy.stats.CensoredData` will result in a\n        ``NotImplementedError``.\n\n    Returns\n    -------\n    res : `~scipy.stats._result_classes.ECDFResult`\n        An object with the following attributes.\n\n        cdf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`\n            An object representing the empirical cumulative distribution\n            function.\n        sf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`\n            An object representing the empirical survival function.\n\n        The `cdf` and `sf` attributes themselves have the following attributes.\n\n        quantiles : ndarray\n            The unique values in the sample that defines the empirical CDF/SF.\n        probabilities : ndarray\n            The point estimates of the probabilities corresponding with\n            `quantiles`.\n\n        And the following methods:\n\n        evaluate(x) :\n            Evaluate the CDF/SF at the argument.\n\n        plot(ax) :\n            Plot the CDF/SF on the provided axes.\n\n        confidence_interval(confidence_level=0.95) :\n            Compute the confidence interval around the CDF/SF at the values in\n            `quantiles`.\n\n    Notes\n    -----\n    When each observation of the sample is a precise measurement, the ECDF\n    steps up by ``1/len(sample)`` at each of the observations [1]_.\n\n    When observations are lower bounds, upper bounds, or both upper and lower\n    bounds, the data is said to be "censored", and `sample` may be provided as\n    an instance of `scipy.stats.CensoredData`.\n\n    For right-censored data, the ECDF is given by the Kaplan-Meier estimator\n    [2]_; other forms of censoring are not supported at this time.\n\n    Confidence intervals are computed according to the Greenwood formula or the\n    more recent "Exponential Greenwood" formula as described in [4]_.\n\n    References\n    ----------\n    .. [1] Conover, William Jay. Practical nonparametric statistics. Vol. 350.\n           John Wiley & Sons, 1999.\n\n    .. [2] Kaplan, Edward L., and Paul Meier. "Nonparametric estimation from\n           incomplete observations." Journal of the American statistical\n           association 53.282 (1958): 457-481.\n\n    .. [3] Goel, Manish Kumar, Pardeep Khanna, and Jugal Kishore.\n           "Understanding survival analysis: Kaplan-Meier estimate."\n           International journal of Ayurveda research 1.4 (2010): 274.\n\n    .. [4] Sawyer, Stanley. "The Greenwood and Exponential Greenwood Confidence\n           Intervals in Survival Analysis."\n           https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf\n\n    Examples\n    --------\n    **Uncensored Data**\n\n    As in the example from [1]_ page 79, five boys were selected at random from\n    those in a single high school. Their one-mile run times were recorded as\n    follows.\n\n    >>> sample = [6.23, 5.58, 7.06, 6.42, 5.20]  # one-mile run times (minutes)\n\n    The empirical distribution function, which approximates the distribution\n    function of one-mile run times of the population from which the boys were\n    sampled, is calculated as follows.\n\n    >>> from scipy import stats\n    >>> res = stats.ecdf(sample)\n    >>> res.cdf.quantiles\n    array([5.2 , 5.58, 6.23, 6.42, 7.06])\n    >>> res.cdf.probabilities\n    array([0.2, 0.4, 0.6, 0.8, 1. ])\n\n    To plot the result as a step function:\n\n    >>> import matplotlib.pyplot as plt\n    >>> ax = plt.subplot()\n    >>> res.cdf.plot(ax)\n    >>> ax.set_xlabel(\'One-Mile Run Time (minutes)\')\n    >>> ax.set_ylabel(\'Empirical CDF\')\n    >>> plt.show()\n\n    **Right-censored Data**\n\n    As in the example from [1]_ page 91, the lives of ten car fanbelts were\n    tested. Five tests concluded because the fanbelt being tested broke, but\n    the remaining tests concluded for other reasons (e.g. the study ran out of\n    funding, but the fanbelt was still functional). The mileage driven\n    with the fanbelts were recorded as follows.\n\n    >>> broken = [77, 47, 81, 56, 80]  # in thousands of miles driven\n    >>> unbroken = [62, 60, 43, 71, 37]\n\n    Precise survival times of the fanbelts that were still functional at the\n    end of the tests are unknown, but they are known to exceed the values\n    recorded in ``unbroken``. Therefore, these observations are said to be\n    "right-censored", and the data is represented using\n    `scipy.stats.CensoredData`.\n\n    >>> sample = stats.CensoredData(uncensored=broken, right=unbroken)\n\n    The empirical survival function is calculated as follows.\n\n    >>> res = stats.ecdf(sample)\n    >>> res.sf.quantiles\n    array([37., 43., 47., 56., 60., 62., 71., 77., 80., 81.])\n    >>> res.sf.probabilities\n    array([1.   , 1.   , 0.875, 0.75 , 0.75 , 0.75 , 0.75 , 0.5  , 0.25 , 0.   ])\n\n    To plot the result as a step function:\n\n    >>> ax = plt.subplot()\n    >>> res.cdf.plot(ax)\n    >>> ax.set_xlabel(\'Fanbelt Survival Time (thousands of miles)\')\n    >>> ax.set_ylabel(\'Empirical SF\')\n    >>> plt.show()\n\n    '
    sample = _iv_CensoredData(sample)
    if sample.num_censored() == 0:
        res = _ecdf_uncensored(sample._uncensor())
    elif sample.num_censored() == sample._right.size:
        res = _ecdf_right_censored(sample)
    else:
        message = 'Currently, only uncensored and right-censored data is supported.'
        raise NotImplementedError(message)
    (t, cdf, sf, n, d) = res
    return ECDFResult(t, cdf, sf, n, d)

def _ecdf_uncensored(sample):
    if False:
        print('Hello World!')
    sample = np.sort(sample)
    (x, counts) = np.unique(sample, return_counts=True)
    events = np.cumsum(counts)
    n = sample.size
    cdf = events / n
    sf = 1 - cdf
    at_risk = np.concatenate(([n], n - events[:-1]))
    return (x, cdf, sf, at_risk, counts)

def _ecdf_right_censored(sample):
    if False:
        i = 10
        return i + 15
    tod = sample._uncensored
    tol = sample._right
    times = np.concatenate((tod, tol))
    died = np.asarray([1] * tod.size + [0] * tol.size)
    i = np.argsort(times)
    times = times[i]
    died = died[i]
    at_risk = np.arange(times.size, 0, -1)
    j = np.diff(times, prepend=-np.inf, append=np.inf) > 0
    j_l = j[:-1]
    j_r = j[1:]
    t = times[j_l]
    n = at_risk[j_l]
    cd = np.cumsum(died)[j_r]
    d = np.diff(cd, prepend=0)
    sf = np.cumprod((n - d) / n)
    cdf = 1 - sf
    return (t, cdf, sf, n, d)

@dataclass
class LogRankResult:
    """Result object returned by `scipy.stats.logrank`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic (defined below). Its magnitude is the
        square root of the magnitude returned by most other logrank test
        implementations.
    pvalue : float ndarray
        The computed p-value of the test.
    """
    statistic: np.ndarray
    pvalue: np.ndarray

def logrank(x: npt.ArrayLike | CensoredData, y: npt.ArrayLike | CensoredData, alternative: Literal['two-sided', 'less', 'greater']='two-sided') -> LogRankResult:
    if False:
        return 10
    'Compare the survival distributions of two samples via the logrank test.\n\n    Parameters\n    ----------\n    x, y : array_like or CensoredData\n        Samples to compare based on their empirical survival functions.\n    alternative : {\'two-sided\', \'less\', \'greater\'}, optional\n        Defines the alternative hypothesis.\n\n        The null hypothesis is that the survival distributions of the two\n        groups, say *X* and *Y*, are identical.\n\n        The following alternative hypotheses [4]_ are available (default is\n        \'two-sided\'):\n\n        * \'two-sided\': the survival distributions of the two groups are not\n          identical.\n        * \'less\': survival of group *X* is favored: the group *X* failure rate\n          function is less than the group *Y* failure rate function at some\n          times.\n        * \'greater\': survival of group *Y* is favored: the group *X* failure\n          rate function is greater than the group *Y* failure rate function at\n          some times.\n\n    Returns\n    -------\n    res : `~scipy.stats._result_classes.LogRankResult`\n        An object containing attributes:\n\n        statistic : float ndarray\n            The computed statistic (defined below). Its magnitude is the\n            square root of the magnitude returned by most other logrank test\n            implementations.\n        pvalue : float ndarray\n            The computed p-value of the test.\n\n    See Also\n    --------\n    scipy.stats.ecdf\n\n    Notes\n    -----\n    The logrank test [1]_ compares the observed number of events to\n    the expected number of events under the null hypothesis that the two\n    samples were drawn from the same distribution. The statistic is\n\n    .. math::\n\n        Z_i = \\frac{\\sum_{j=1}^J(O_{i,j}-E_{i,j})}{\\sqrt{\\sum_{j=1}^J V_{i,j}}}\n        \\rightarrow \\mathcal{N}(0,1)\n\n    where\n\n    .. math::\n\n        E_{i,j} = O_j \\frac{N_{i,j}}{N_j},\n        \\qquad\n        V_{i,j} = E_{i,j} \\left(\\frac{N_j-O_j}{N_j}\\right)\n        \\left(\\frac{N_j-N_{i,j}}{N_j-1}\\right),\n\n    :math:`i` denotes the group (i.e. it may assume values :math:`x` or\n    :math:`y`, or it may be omitted to refer to the combined sample)\n    :math:`j` denotes the time (at which an event occurred),\n    :math:`N` is the number of subjects at risk just before an event occurred,\n    and :math:`O` is the observed number of events at that time.\n\n    The ``statistic`` :math:`Z_x` returned by `logrank` is the (signed) square\n    root of the statistic returned by many other implementations. Under the\n    null hypothesis, :math:`Z_x**2` is asymptotically distributed according to\n    the chi-squared distribution with one degree of freedom. Consequently,\n    :math:`Z_x` is asymptotically distributed according to the standard normal\n    distribution. The advantage of using :math:`Z_x` is that the sign\n    information (i.e. whether the observed number of events tends to be less\n    than or greater than the number expected under the null hypothesis) is\n    preserved, allowing `scipy.stats.logrank` to offer one-sided alternative\n    hypotheses.\n\n    References\n    ----------\n    .. [1] Mantel N. "Evaluation of survival data and two new rank order\n           statistics arising in its consideration."\n           Cancer Chemotherapy Reports, 50(3):163-170, PMID: 5910392, 1966\n    .. [2] Bland, Altman, "The logrank test", BMJ, 328:1073,\n           :doi:`10.1136/bmj.328.7447.1073`, 2004\n    .. [3] "Logrank test", Wikipedia,\n           https://en.wikipedia.org/wiki/Logrank_test\n    .. [4] Brown, Mark. "On the choice of variance for the log rank test."\n           Biometrika 71.1 (1984): 65-74.\n    .. [5] Klein, John P., and Melvin L. Moeschberger. Survival analysis:\n           techniques for censored and truncated data. Vol. 1230. New York:\n           Springer, 2003.\n\n    Examples\n    --------\n    Reference [2]_ compared the survival times of patients with two different\n    types of recurrent malignant gliomas. The samples below record the time\n    (number of weeks) for which each patient participated in the study. The\n    `scipy.stats.CensoredData` class is used because the data is\n    right-censored: the uncensored observations correspond with observed deaths\n    whereas the censored observations correspond with the patient leaving the\n    study for another reason.\n\n    >>> from scipy import stats\n    >>> x = stats.CensoredData(\n    ...     uncensored=[6, 13, 21, 30, 37, 38, 49, 50,\n    ...                 63, 79, 86, 98, 202, 219],\n    ...     right=[31, 47, 80, 82, 82, 149]\n    ... )\n    >>> y = stats.CensoredData(\n    ...     uncensored=[10, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 24,\n    ...                 25, 28,30, 33, 35, 37, 40, 40, 46, 48, 76, 81,\n    ...                 82, 91, 112, 181],\n    ...     right=[34, 40, 70]\n    ... )\n\n    We can calculate and visualize the empirical survival functions\n    of both groups as follows.\n\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> ax = plt.subplot()\n    >>> ecdf_x = stats.ecdf(x)\n    >>> ecdf_x.sf.plot(ax, label=\'Astrocytoma\')\n    >>> ecdf_y = stats.ecdf(y)\n    >>> ecdf_x.sf.plot(ax, label=\'Glioblastoma\')\n    >>> ax.set_xlabel(\'Time to death (weeks)\')\n    >>> ax.set_ylabel(\'Empirical SF\')\n    >>> plt.legend()\n    >>> plt.show()\n\n    Visual inspection of the empirical survival functions suggests that the\n    survival times tend to be different between the two groups. To formally\n    assess whether the difference is significant at the 1% level, we use the\n    logrank test.\n\n    >>> res = stats.logrank(x=x, y=y)\n    >>> res.statistic\n    -2.73799...\n    >>> res.pvalue\n    0.00618...\n\n    The p-value is less than 1%, so we can consider the data to be evidence\n    against the null hypothesis in favor of the alternative that there is a\n    difference between the two survival functions.\n\n    '
    x = _iv_CensoredData(sample=x, param_name='x')
    y = _iv_CensoredData(sample=y, param_name='y')
    xy = CensoredData(uncensored=np.concatenate((x._uncensored, y._uncensored)), right=np.concatenate((x._right, y._right)))
    res = ecdf(xy)
    idx = res.sf._d.astype(bool)
    times_xy = res.sf.quantiles[idx]
    at_risk_xy = res.sf._n[idx]
    deaths_xy = res.sf._d[idx]
    res_x = ecdf(x)
    i = np.searchsorted(res_x.sf.quantiles, times_xy)
    at_risk_x = np.append(res_x.sf._n, 0)[i]
    at_risk_y = at_risk_xy - at_risk_x
    num = at_risk_x * at_risk_y * deaths_xy * (at_risk_xy - deaths_xy)
    den = at_risk_xy ** 2 * (at_risk_xy - 1)
    i = at_risk_xy > 1
    sum_var = np.sum(num[i] / den[i])
    n_died_x = x._uncensored.size
    sum_exp_deaths_x = np.sum(at_risk_x * (deaths_xy / at_risk_xy))
    statistic = (n_died_x - sum_exp_deaths_x) / np.sqrt(sum_var)
    (_, pvalue) = stats._stats_py._normtest_finish(z=statistic, alternative=alternative)
    return LogRankResult(statistic=statistic, pvalue=pvalue)