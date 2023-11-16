"""Anova k-sample comparison without and with trimming

Created on Sun Jun 09 23:51:34 2013

Author: Josef Perktold
"""
import numbers
import numpy as np

def trimboth(a, proportiontocut, axis=0):
    if False:
        i = 10
        return i + 15
    "\n    Slices off a proportion of items from both ends of an array.\n\n    Slices off the passed proportion of items from both ends of the passed\n    array (i.e., with `proportiontocut` = 0.1, slices leftmost 10% **and**\n    rightmost 10% of scores).  You must pre-sort the array if you want\n    'proper' trimming.  Slices off less if proportion results in a\n    non-integer slice index (i.e., conservatively slices off\n    `proportiontocut`).\n\n    Parameters\n    ----------\n    a : array_like\n        Data to trim.\n    proportiontocut : float or int\n        Proportion of data to trim at each end.\n    axis : int or None\n        Axis along which the observations are trimmed. The default is to trim\n        along axis=0. If axis is None then the array will be flattened before\n        trimming.\n\n    Returns\n    -------\n    out : array-like\n        Trimmed version of array `a`.\n\n    Examples\n    --------\n    >>> from scipy import stats\n    >>> a = np.arange(20)\n    >>> b = stats.trimboth(a, 0.1)\n    >>> b.shape\n    (16,)\n\n    "
    a = np.asarray(a)
    if axis is None:
        a = a.ravel()
        axis = 0
    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut >= uppercut:
        raise ValueError('Proportion too big.')
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(lowercut, uppercut)
    return a[tuple(sl)]

def trim_mean(a, proportiontocut, axis=0):
    if False:
        while True:
            i = 10
    "\n    Return mean of array after trimming observations from both tails.\n\n    If `proportiontocut` = 0.1, slices off 'leftmost' and 'rightmost' 10% of\n    scores. Slices off LESS if proportion results in a non-integer slice\n    index (i.e., conservatively slices off `proportiontocut` ).\n\n    Parameters\n    ----------\n    a : array_like\n        Input array\n    proportiontocut : float\n        Fraction to cut off at each tail of the sorted observations.\n    axis : int or None\n        Axis along which the trimmed means are computed. The default is axis=0.\n        If axis is None then the trimmed mean will be computed for the\n        flattened array.\n\n    Returns\n    -------\n    trim_mean : ndarray\n        Mean of trimmed array.\n\n    "
    newa = trimboth(np.sort(a, axis), proportiontocut, axis=axis)
    return np.mean(newa, axis=axis)

class TrimmedMean:
    """
    class for trimmed and winsorized one sample statistics

    axis is None, i.e. ravelling, is not supported

    Parameters
    ----------
    data : array-like
        The data, observations to analyze.
    fraction : float in (0, 0.5)
        The fraction of observations to trim at each tail.
        The number of observations trimmed at each tail is
        ``int(fraction * nobs)``
    is_sorted : boolean
        Indicator if data is already sorted. By default the data is sorted
        along ``axis``.
    axis : int
        The axis of reduce operations. By default axis=0, that is observations
        are along the zero dimension, i.e. rows if 2-dim.
    """

    def __init__(self, data, fraction, is_sorted=False, axis=0):
        if False:
            i = 10
            return i + 15
        self.data = np.asarray(data)
        self.axis = axis
        self.fraction = fraction
        self.nobs = nobs = self.data.shape[axis]
        self.lowercut = lowercut = int(fraction * nobs)
        self.uppercut = uppercut = nobs - lowercut
        if lowercut >= uppercut:
            raise ValueError('Proportion too big.')
        self.nobs_reduced = nobs - 2 * lowercut
        self.sl = [slice(None)] * self.data.ndim
        self.sl[axis] = slice(self.lowercut, self.uppercut)
        self.sl = tuple(self.sl)
        if not is_sorted:
            self.data_sorted = np.sort(self.data, axis=axis)
        else:
            self.data_sorted = self.data
        self.lowerbound = np.take(self.data_sorted, lowercut, axis=axis)
        self.upperbound = np.take(self.data_sorted, uppercut - 1, axis=axis)

    @property
    def data_trimmed(self):
        if False:
            return 10
        'numpy array of trimmed and sorted data\n        '
        return self.data_sorted[self.sl]

    @property
    def data_winsorized(self):
        if False:
            return 10
        'winsorized data\n        '
        lb = np.expand_dims(self.lowerbound, self.axis)
        ub = np.expand_dims(self.upperbound, self.axis)
        return np.clip(self.data_sorted, lb, ub)

    @property
    def mean_trimmed(self):
        if False:
            return 10
        'mean of trimmed data\n        '
        return np.mean(self.data_sorted[tuple(self.sl)], self.axis)

    @property
    def mean_winsorized(self):
        if False:
            while True:
                i = 10
        'mean of winsorized data\n        '
        return np.mean(self.data_winsorized, self.axis)

    @property
    def var_winsorized(self):
        if False:
            print('Hello World!')
        'variance of winsorized data\n        '
        return np.var(self.data_winsorized, ddof=1, axis=self.axis)

    @property
    def std_mean_trimmed(self):
        if False:
            return 10
        'standard error of trimmed mean\n        '
        se = np.sqrt(self.var_winsorized / self.nobs_reduced)
        se *= np.sqrt(self.nobs / self.nobs_reduced)
        return se

    @property
    def std_mean_winsorized(self):
        if False:
            while True:
                i = 10
        'standard error of winsorized mean\n        '
        std_ = np.sqrt(self.var_winsorized / self.nobs)
        std_ *= (self.nobs - 1) / (self.nobs_reduced - 1)
        return std_

    def ttest_mean(self, value=0, transform='trimmed', alternative='two-sided'):
        if False:
            print('Hello World!')
        "\n        One sample t-test for trimmed or Winsorized mean\n\n        Parameters\n        ----------\n        value : float\n            Value of the mean under the Null hypothesis\n        transform : {'trimmed', 'winsorized'}\n            Specified whether the mean test is based on trimmed or winsorized\n            data.\n        alternative : {'two-sided', 'larger', 'smaller'}\n\n\n        Notes\n        -----\n        p-value is based on the approximate t-distribution of the test\n        statistic. The approximation is valid if the underlying distribution\n        is symmetric.\n        "
        import statsmodels.stats.weightstats as smws
        df = self.nobs_reduced - 1
        if transform == 'trimmed':
            mean_ = self.mean_trimmed
            std_ = self.std_mean_trimmed
        elif transform == 'winsorized':
            mean_ = self.mean_winsorized
            std_ = self.std_mean_winsorized
        else:
            raise ValueError("transform can only be 'trimmed' or 'winsorized'")
        res = smws._tstat_generic(mean_, 0, std_, df, alternative=alternative, diff=value)
        return res + (df,)

    def reset_fraction(self, frac):
        if False:
            i = 10
            return i + 15
        'create a TrimmedMean instance with a new trimming fraction\n\n        This reuses the sorted array from the current instance.\n        '
        tm = TrimmedMean(self.data_sorted, frac, is_sorted=True, axis=self.axis)
        tm.data = self.data
        return tm

def scale_transform(data, center='median', transform='abs', trim_frac=0.2, axis=0):
    if False:
        while True:
            i = 10
    'Transform data for variance comparison for Levene type tests\n\n    Parameters\n    ----------\n    data : array_like\n        Observations for the data.\n    center : "median", "mean", "trimmed" or float\n        Statistic used for centering observations. If a float, then this\n        value is used to center. Default is median.\n    transform : \'abs\', \'square\', \'identity\' or a callable\n        The transform for the centered data.\n    trim_frac : float in [0, 0.5)\n        Fraction of observations that are trimmed on each side of the sorted\n        observations. This is only used if center is `trimmed`.\n    axis : int\n        Axis along which the data are transformed when centering.\n\n    Returns\n    -------\n    res : ndarray\n        transformed data in the same shape as the original data.\n\n    '
    x = np.asarray(data)
    if transform == 'abs':
        tfunc = np.abs
    elif transform == 'square':
        tfunc = lambda x: x * x
    elif transform == 'identity':
        tfunc = lambda x: x
    elif callable(transform):
        tfunc = transform
    else:
        raise ValueError('transform should be abs, square or exp')
    if center == 'median':
        res = tfunc(x - np.expand_dims(np.median(x, axis=axis), axis))
    elif center == 'mean':
        res = tfunc(x - np.expand_dims(np.mean(x, axis=axis), axis))
    elif center == 'trimmed':
        center = trim_mean(x, trim_frac, axis=axis)
        res = tfunc(x - np.expand_dims(center, axis))
    elif isinstance(center, numbers.Number):
        res = tfunc(x - center)
    else:
        raise ValueError('center should be median, mean or trimmed')
    return res