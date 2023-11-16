"""
An extension of scipy.stats._stats_py to support masked arrays

"""
__all__ = ['argstoarray', 'count_tied_groups', 'describe', 'f_oneway', 'find_repeats', 'friedmanchisquare', 'kendalltau', 'kendalltau_seasonal', 'kruskal', 'kruskalwallis', 'ks_twosamp', 'ks_2samp', 'kurtosis', 'kurtosistest', 'ks_1samp', 'kstest', 'linregress', 'mannwhitneyu', 'meppf', 'mode', 'moment', 'mquantiles', 'msign', 'normaltest', 'obrientransform', 'pearsonr', 'plotting_positions', 'pointbiserialr', 'rankdata', 'scoreatpercentile', 'sem', 'sen_seasonal_slopes', 'skew', 'skewtest', 'spearmanr', 'siegelslopes', 'theilslopes', 'tmax', 'tmean', 'tmin', 'trim', 'trimboth', 'trimtail', 'trima', 'trimr', 'trimmed_mean', 'trimmed_std', 'trimmed_stde', 'trimmed_var', 'tsem', 'ttest_1samp', 'ttest_onesamp', 'ttest_ind', 'ttest_rel', 'tvar', 'variation', 'winsorize', 'brunnermunzel']
import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import _find_repeats, linregress as stats_linregress, LinregressResult as stats_LinregressResult, theilslopes as stats_theilslopes, siegelslopes as stats_siegelslopes

def _chk_asarray(a, axis):
    if False:
        for i in range(10):
            print('nop')
    a = ma.asanyarray(a)
    if axis is None:
        a = ma.ravel(a)
        outaxis = 0
    else:
        outaxis = axis
    return (a, outaxis)

def _chk2_asarray(a, b, axis):
    if False:
        i = 10
        return i + 15
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    if axis is None:
        a = ma.ravel(a)
        b = ma.ravel(b)
        outaxis = 0
    else:
        outaxis = axis
    return (a, b, outaxis)

def _chk_size(a, b):
    if False:
        while True:
            i = 10
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    (na, nb) = (a.size, b.size)
    if na != nb:
        raise ValueError('The size of the input array should match! ({} <> {})'.format(na, nb))
    return (a, b, na)

def argstoarray(*args):
    if False:
        print('Hello World!')
    '\n    Constructs a 2D array from a group of sequences.\n\n    Sequences are filled with missing values to match the length of the longest\n    sequence.\n\n    Parameters\n    ----------\n    *args : sequences\n        Group of sequences.\n\n    Returns\n    -------\n    argstoarray : MaskedArray\n        A ( `m` x `n` ) masked array, where `m` is the number of arguments and\n        `n` the length of the longest argument.\n\n    Notes\n    -----\n    `numpy.ma.vstack` has identical behavior, but is called with a sequence\n    of sequences.\n\n    Examples\n    --------\n    A 2D masked array constructed from a group of sequences is returned.\n\n    >>> from scipy.stats.mstats import argstoarray\n    >>> argstoarray([1, 2, 3], [4, 5, 6])\n    masked_array(\n     data=[[1.0, 2.0, 3.0],\n           [4.0, 5.0, 6.0]],\n     mask=[[False, False, False],\n           [False, False, False]],\n     fill_value=1e+20)\n\n    The returned masked array filled with missing values when the lengths of\n    sequences are different.\n\n    >>> argstoarray([1, 3], [4, 5, 6])\n    masked_array(\n     data=[[1.0, 3.0, --],\n           [4.0, 5.0, 6.0]],\n     mask=[[False, False,  True],\n           [False, False, False]],\n     fill_value=1e+20)\n\n    '
    if len(args) == 1 and (not isinstance(args[0], ndarray)):
        output = ma.asarray(args[0])
        if output.ndim != 2:
            raise ValueError('The input should be 2D')
    else:
        n = len(args)
        m = max([len(k) for k in args])
        output = ma.array(np.empty((n, m), dtype=float), mask=True)
        for (k, v) in enumerate(args):
            output[k, :len(v)] = v
    output[np.logical_not(np.isfinite(output._data))] = masked
    return output

def find_repeats(arr):
    if False:
        while True:
            i = 10
    'Find repeats in arr and return a tuple (repeats, repeat_count).\n\n    The input is cast to float64. Masked values are discarded.\n\n    Parameters\n    ----------\n    arr : sequence\n        Input array. The array is flattened if it is not 1D.\n\n    Returns\n    -------\n    repeats : ndarray\n        Array of repeated values.\n    counts : ndarray\n        Array of counts.\n\n    Examples\n    --------\n    >>> from scipy.stats import mstats\n    >>> mstats.find_repeats([2, 1, 2, 3, 2, 2, 5])\n    (array([2.]), array([4]))\n\n    In the above example, 2 repeats 4 times.\n\n    >>> mstats.find_repeats([[10, 20, 1, 2], [5, 5, 4, 4]])\n    (array([4., 5.]), array([2, 2]))\n\n    In the above example, both 4 and 5 repeat 2 times.\n\n    '
    compr = np.asarray(ma.compressed(arr), dtype=np.float64)
    try:
        need_copy = np.may_share_memory(compr, arr)
    except AttributeError:
        need_copy = False
    if need_copy:
        compr = compr.copy()
    return _find_repeats(compr)

def count_tied_groups(x, use_missing=False):
    if False:
        return 10
    '\n    Counts the number of tied values.\n\n    Parameters\n    ----------\n    x : sequence\n        Sequence of data on which to counts the ties\n    use_missing : bool, optional\n        Whether to consider missing values as tied.\n\n    Returns\n    -------\n    count_tied_groups : dict\n        Returns a dictionary (nb of ties: nb of groups).\n\n    Examples\n    --------\n    >>> from scipy.stats import mstats\n    >>> import numpy as np\n    >>> z = [0, 0, 0, 2, 2, 2, 3, 3, 4, 5, 6]\n    >>> mstats.count_tied_groups(z)\n    {2: 1, 3: 2}\n\n    In the above example, the ties were 0 (3x), 2 (3x) and 3 (2x).\n\n    >>> z = np.ma.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 6])\n    >>> mstats.count_tied_groups(z)\n    {2: 2, 3: 1}\n    >>> z[[1,-1]] = np.ma.masked\n    >>> mstats.count_tied_groups(z, use_missing=True)\n    {2: 2, 3: 1}\n\n    '
    nmasked = ma.getmask(x).sum()
    data = ma.compressed(x).copy()
    (ties, counts) = find_repeats(data)
    nties = {}
    if len(ties):
        nties = dict(zip(np.unique(counts), itertools.repeat(1)))
        nties.update(dict(zip(*find_repeats(counts))))
    if nmasked and use_missing:
        try:
            nties[nmasked] += 1
        except KeyError:
            nties[nmasked] = 1
    return nties

def rankdata(data, axis=None, use_missing=False):
    if False:
        return 10
    'Returns the rank (also known as order statistics) of each data point\n    along the given axis.\n\n    If some values are tied, their rank is averaged.\n    If some values are masked, their rank is set to 0 if use_missing is False,\n    or set to the average rank of the unmasked values if use_missing is True.\n\n    Parameters\n    ----------\n    data : sequence\n        Input data. The data is transformed to a masked array\n    axis : {None,int}, optional\n        Axis along which to perform the ranking.\n        If None, the array is first flattened. An exception is raised if\n        the axis is specified for arrays with a dimension larger than 2\n    use_missing : bool, optional\n        Whether the masked values have a rank of 0 (False) or equal to the\n        average rank of the unmasked values (True).\n\n    '

    def _rank1d(data, use_missing=False):
        if False:
            print('Hello World!')
        n = data.count()
        rk = np.empty(data.size, dtype=float)
        idx = data.argsort()
        rk[idx[:n]] = np.arange(1, n + 1)
        if use_missing:
            rk[idx[n:]] = (n + 1) / 2.0
        else:
            rk[idx[n:]] = 0
        repeats = find_repeats(data.copy())
        for r in repeats[0]:
            condition = (data == r).filled(False)
            rk[condition] = rk[condition].mean()
        return rk
    data = ma.array(data, copy=False)
    if axis is None:
        if data.ndim > 1:
            return _rank1d(data.ravel(), use_missing).reshape(data.shape)
        else:
            return _rank1d(data, use_missing)
    else:
        return ma.apply_along_axis(_rank1d, axis, data, use_missing).view(ndarray)
ModeResult = namedtuple('ModeResult', ('mode', 'count'))

def mode(a, axis=0):
    if False:
        return 10
    '\n    Returns an array of the modal (most common) value in the passed array.\n\n    Parameters\n    ----------\n    a : array_like\n        n-dimensional array of which to find mode(s).\n    axis : int or None, optional\n        Axis along which to operate. Default is 0. If None, compute over\n        the whole array `a`.\n\n    Returns\n    -------\n    mode : ndarray\n        Array of modal values.\n    count : ndarray\n        Array of counts for each mode.\n\n    Notes\n    -----\n    For more details, see `scipy.stats.mode`.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import stats\n    >>> from scipy.stats import mstats\n    >>> m_arr = np.ma.array([1, 1, 0, 0, 0, 0], mask=[0, 0, 1, 1, 1, 0])\n    >>> mstats.mode(m_arr)  # note that most zeros are masked\n    ModeResult(mode=array([1.]), count=array([2.]))\n\n    '
    return _mode(a, axis=axis, keepdims=True)

def _mode(a, axis=0, keepdims=True):
    if False:
        for i in range(10):
            print('nop')
    (a, axis) = _chk_asarray(a, axis)

    def _mode1D(a):
        if False:
            for i in range(10):
                print('nop')
        (rep, cnt) = find_repeats(a)
        if not cnt.ndim:
            return (0, 0)
        elif cnt.size:
            return (rep[cnt.argmax()], cnt.max())
        else:
            return (a.min(), 1)
    if axis is None:
        output = _mode1D(ma.ravel(a))
        output = (ma.array(output[0]), ma.array(output[1]))
    else:
        output = ma.apply_along_axis(_mode1D, axis, a)
        if keepdims is None or keepdims:
            newshape = list(a.shape)
            newshape[axis] = 1
            slices = [slice(None)] * output.ndim
            slices[axis] = 0
            modes = output[tuple(slices)].reshape(newshape)
            slices[axis] = 1
            counts = output[tuple(slices)].reshape(newshape)
            output = (modes, counts)
        else:
            output = np.moveaxis(output, axis, 0)
    return ModeResult(*output)

def _betai(a, b, x):
    if False:
        return 10
    x = np.asanyarray(x)
    x = ma.where(x < 1.0, x, 1.0)
    return special.betainc(a, b, x)

def msign(x):
    if False:
        return 10
    'Returns the sign of x, or 0 if x is masked.'
    return ma.filled(np.sign(x), 0)

def pearsonr(x, y):
    if False:
        while True:
            i = 10
    '\n    Pearson correlation coefficient and p-value for testing non-correlation.\n\n    The Pearson correlation coefficient [1]_ measures the linear relationship\n    between two datasets.  The calculation of the p-value relies on the\n    assumption that each dataset is normally distributed.  (See Kowalski [3]_\n    for a discussion of the effects of non-normality of the input on the\n    distribution of the correlation coefficient.)  Like other correlation\n    coefficients, this one varies between -1 and +1 with 0 implying no\n    correlation. Correlations of -1 or +1 imply an exact linear relationship.\n\n    Parameters\n    ----------\n    x : (N,) array_like\n        Input array.\n    y : (N,) array_like\n        Input array.\n\n    Returns\n    -------\n    r : float\n        Pearson\'s correlation coefficient.\n    p-value : float\n        Two-tailed p-value.\n\n    Warns\n    -----\n    `~scipy.stats.ConstantInputWarning`\n        Raised if an input is a constant array.  The correlation coefficient\n        is not defined in this case, so ``np.nan`` is returned.\n\n    `~scipy.stats.NearConstantInputWarning`\n        Raised if an input is "nearly" constant.  The array ``x`` is considered\n        nearly constant if ``norm(x - mean(x)) < 1e-13 * abs(mean(x))``.\n        Numerical errors in the calculation ``x - mean(x)`` in this case might\n        result in an inaccurate calculation of r.\n\n    See Also\n    --------\n    spearmanr : Spearman rank-order correlation coefficient.\n    kendalltau : Kendall\'s tau, a correlation measure for ordinal data.\n\n    Notes\n    -----\n    The correlation coefficient is calculated as follows:\n\n    .. math::\n\n        r = \\frac{\\sum (x - m_x) (y - m_y)}\n                 {\\sqrt{\\sum (x - m_x)^2 \\sum (y - m_y)^2}}\n\n    where :math:`m_x` is the mean of the vector x and :math:`m_y` is\n    the mean of the vector y.\n\n    Under the assumption that x and y are drawn from\n    independent normal distributions (so the population correlation coefficient\n    is 0), the probability density function of the sample correlation\n    coefficient r is ([1]_, [2]_):\n\n    .. math::\n\n        f(r) = \\frac{{(1-r^2)}^{n/2-2}}{\\mathrm{B}(\\frac{1}{2},\\frac{n}{2}-1)}\n\n    where n is the number of samples, and B is the beta function.  This\n    is sometimes referred to as the exact distribution of r.  This is\n    the distribution that is used in `pearsonr` to compute the p-value.\n    The distribution is a beta distribution on the interval [-1, 1],\n    with equal shape parameters a = b = n/2 - 1.  In terms of SciPy\'s\n    implementation of the beta distribution, the distribution of r is::\n\n        dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)\n\n    The p-value returned by `pearsonr` is a two-sided p-value. The p-value\n    roughly indicates the probability of an uncorrelated system\n    producing datasets that have a Pearson correlation at least as extreme\n    as the one computed from these datasets. More precisely, for a\n    given sample with correlation coefficient r, the p-value is\n    the probability that abs(r\') of a random sample x\' and y\' drawn from\n    the population with zero correlation would be greater than or equal\n    to abs(r). In terms of the object ``dist`` shown above, the p-value\n    for a given r and length n can be computed as::\n\n        p = 2*dist.cdf(-abs(r))\n\n    When n is 2, the above continuous distribution is not well-defined.\n    One can interpret the limit of the beta distribution as the shape\n    parameters a and b approach a = b = 0 as a discrete distribution with\n    equal probability masses at r = 1 and r = -1.  More directly, one\n    can observe that, given the data x = [x1, x2] and y = [y1, y2], and\n    assuming x1 != x2 and y1 != y2, the only possible values for r are 1\n    and -1.  Because abs(r\') for any sample x\' and y\' with length 2 will\n    be 1, the two-sided p-value for a sample of length 2 is always 1.\n\n    References\n    ----------\n    .. [1] "Pearson correlation coefficient", Wikipedia,\n           https://en.wikipedia.org/wiki/Pearson_correlation_coefficient\n    .. [2] Student, "Probable error of a correlation coefficient",\n           Biometrika, Volume 6, Issue 2-3, 1 September 1908, pp. 302-310.\n    .. [3] C. J. Kowalski, "On the Effects of Non-Normality on the Distribution\n           of the Sample Product-Moment Correlation Coefficient"\n           Journal of the Royal Statistical Society. Series C (Applied\n           Statistics), Vol. 21, No. 1 (1972), pp. 1-12.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import stats\n    >>> from scipy.stats import mstats\n    >>> mstats.pearsonr([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4])\n    (-0.7426106572325057, 0.1505558088534455)\n\n    There is a linear dependence between x and y if y = a + b*x + e, where\n    a,b are constants and e is a random error term, assumed to be independent\n    of x. For simplicity, assume that x is standard normal, a=0, b=1 and let\n    e follow a normal distribution with mean zero and standard deviation s>0.\n\n    >>> s = 0.5\n    >>> x = stats.norm.rvs(size=500)\n    >>> e = stats.norm.rvs(scale=s, size=500)\n    >>> y = x + e\n    >>> mstats.pearsonr(x, y)\n    (0.9029601878969703, 8.428978827629898e-185) # may vary\n\n    This should be close to the exact value given by\n\n    >>> 1/np.sqrt(1 + s**2)\n    0.8944271909999159\n\n    For s=0.5, we observe a high level of correlation. In general, a large\n    variance of the noise reduces the correlation, while the correlation\n    approaches one as the variance of the error goes to zero.\n\n    It is important to keep in mind that no correlation does not imply\n    independence unless (x, y) is jointly normal. Correlation can even be zero\n    when there is a very simple dependence structure: if X follows a\n    standard normal distribution, let y = abs(x). Note that the correlation\n    between x and y is zero. Indeed, since the expectation of x is zero,\n    cov(x, y) = E[x*y]. By definition, this equals E[x*abs(x)] which is zero\n    by symmetry. The following lines of code illustrate this observation:\n\n    >>> y = np.abs(x)\n    >>> mstats.pearsonr(x, y)\n    (-0.016172891856853524, 0.7182823678751942) # may vary\n\n    A non-zero correlation coefficient can be misleading. For example, if X has\n    a standard normal distribution, define y = x if x < 0 and y = 0 otherwise.\n    A simple calculation shows that corr(x, y) = sqrt(2/Pi) = 0.797...,\n    implying a high level of correlation:\n\n    >>> y = np.where(x < 0, x, 0)\n    >>> mstats.pearsonr(x, y)\n    (0.8537091583771509, 3.183461621422181e-143) # may vary\n\n    This is unintuitive since there is no dependence of x and y if x is larger\n    than zero which happens in about half of the cases if we sample x and y.\n    '
    (x, y, n) = _chk_size(x, y)
    (x, y) = (x.ravel(), y.ravel())
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    n -= m.sum()
    df = n - 2
    if df < 0:
        return (masked, masked)
    return scipy.stats._stats_py.pearsonr(ma.masked_array(x, mask=m).compressed(), ma.masked_array(y, mask=m).compressed())

def spearmanr(x, y=None, use_ties=True, axis=None, nan_policy='propagate', alternative='two-sided'):
    if False:
        i = 10
        return i + 15
    "\n    Calculates a Spearman rank-order correlation coefficient and the p-value\n    to test for non-correlation.\n\n    The Spearman correlation is a nonparametric measure of the linear\n    relationship between two datasets. Unlike the Pearson correlation, the\n    Spearman correlation does not assume that both datasets are normally\n    distributed. Like other correlation coefficients, this one varies\n    between -1 and +1 with 0 implying no correlation. Correlations of -1 or\n    +1 imply a monotonic relationship. Positive correlations imply that\n    as `x` increases, so does `y`. Negative correlations imply that as `x`\n    increases, `y` decreases.\n\n    Missing values are discarded pair-wise: if a value is missing in `x`, the\n    corresponding value in `y` is masked.\n\n    The p-value roughly indicates the probability of an uncorrelated system\n    producing datasets that have a Spearman correlation at least as extreme\n    as the one computed from these datasets. The p-values are not entirely\n    reliable but are probably reasonable for datasets larger than 500 or so.\n\n    Parameters\n    ----------\n    x, y : 1D or 2D array_like, y is optional\n        One or two 1-D or 2-D arrays containing multiple variables and\n        observations. When these are 1-D, each represents a vector of\n        observations of a single variable. For the behavior in the 2-D case,\n        see under ``axis``, below.\n    use_ties : bool, optional\n        DO NOT USE.  Does not do anything, keyword is only left in place for\n        backwards compatibility reasons.\n    axis : int or None, optional\n        If axis=0 (default), then each column represents a variable, with\n        observations in the rows. If axis=1, the relationship is transposed:\n        each row represents a variable, while the columns contain observations.\n        If axis=None, then both arrays will be raveled.\n    nan_policy : {'propagate', 'raise', 'omit'}, optional\n        Defines how to handle when input contains nan. 'propagate' returns nan,\n        'raise' throws an error, 'omit' performs the calculations ignoring nan\n        values. Default is 'propagate'.\n    alternative : {'two-sided', 'less', 'greater'}, optional\n        Defines the alternative hypothesis. Default is 'two-sided'.\n        The following options are available:\n\n        * 'two-sided': the correlation is nonzero\n        * 'less': the correlation is negative (less than zero)\n        * 'greater':  the correlation is positive (greater than zero)\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    res : SignificanceResult\n        An object containing attributes:\n\n        statistic : float or ndarray (2-D square)\n            Spearman correlation matrix or correlation coefficient (if only 2\n            variables are given as parameters). Correlation matrix is square\n            with length equal to total number of variables (columns or rows) in\n            ``a`` and ``b`` combined.\n        pvalue : float\n            The p-value for a hypothesis test whose null hypothesis\n            is that two sets of data are linearly uncorrelated. See\n            `alternative` above for alternative hypotheses. `pvalue` has the\n            same shape as `statistic`.\n\n    References\n    ----------\n    [CRCProbStat2000] section 14.7\n\n    "
    if not use_ties:
        raise ValueError('`use_ties=False` is not supported in SciPy >= 1.2.0')
    (x, axisout) = _chk_asarray(x, axis)
    if y is not None:
        (y, _) = _chk_asarray(y, axis)
        if axisout == 0:
            x = ma.column_stack((x, y))
        else:
            x = ma.vstack((x, y))
    if axisout == 1:
        x = x.T
    if nan_policy == 'omit':
        x = ma.masked_invalid(x)

    def _spearmanr_2cols(x):
        if False:
            i = 10
            return i + 15
        x = ma.mask_rowcols(x, axis=0)
        x = x[~x.mask.any(axis=1), :]
        if not np.any(x.data):
            res = scipy.stats._stats_py.SignificanceResult(np.nan, np.nan)
            res.correlation = np.nan
            return res
        m = ma.getmask(x)
        n_obs = x.shape[0]
        dof = n_obs - 2 - int(m.sum(axis=0)[0])
        if dof < 0:
            raise ValueError('The input must have at least 3 entries!')
        x_ranked = rankdata(x, axis=0)
        rs = ma.corrcoef(x_ranked, rowvar=False).data
        with np.errstate(divide='ignore'):
            t = rs * np.sqrt((dof / ((rs + 1.0) * (1.0 - rs))).clip(0))
        (t, prob) = scipy.stats._stats_py._ttest_finish(dof, t, alternative)
        if rs.shape == (2, 2):
            res = scipy.stats._stats_py.SignificanceResult(rs[1, 0], prob[1, 0])
            res.correlation = rs[1, 0]
            return res
        else:
            res = scipy.stats._stats_py.SignificanceResult(rs, prob)
            res.correlation = rs
            return res
    n_vars = x.shape[1]
    if n_vars == 2:
        return _spearmanr_2cols(x)
    else:
        rs = np.ones((n_vars, n_vars), dtype=float)
        prob = np.zeros((n_vars, n_vars), dtype=float)
        for var1 in range(n_vars - 1):
            for var2 in range(var1 + 1, n_vars):
                result = _spearmanr_2cols(x[:, [var1, var2]])
                rs[var1, var2] = result.correlation
                rs[var2, var1] = result.correlation
                prob[var1, var2] = result.pvalue
                prob[var2, var1] = result.pvalue
        res = scipy.stats._stats_py.SignificanceResult(rs, prob)
        res.correlation = rs
        return res

def _kendall_p_exact(n, c, alternative='two-sided'):
    if False:
        return 10
    in_right_tail = c >= n * (n - 1) // 2 - c
    alternative_greater = alternative == 'greater'
    c = int(min(c, n * (n - 1) // 2 - c))
    if n <= 0:
        raise ValueError(f'n ({n}) must be positive')
    elif c < 0 or 4 * c > n * (n - 1):
        raise ValueError(f'c ({c}) must satisfy 0 <= 4c <= n(n-1) = {n * (n - 1)}.')
    elif n == 1:
        prob = 1.0
        p_mass_at_c = 1
    elif n == 2:
        prob = 1.0
        p_mass_at_c = 0.5
    elif c == 0:
        prob = 2.0 / math.factorial(n) if n < 171 else 0.0
        p_mass_at_c = prob / 2
    elif c == 1:
        prob = 2.0 / math.factorial(n - 1) if n < 172 else 0.0
        p_mass_at_c = (n - 1) / math.factorial(n)
    elif 4 * c == n * (n - 1) and alternative == 'two-sided':
        prob = 1.0
    elif n < 171:
        new = np.zeros(c + 1)
        new[0:2] = 1.0
        for j in range(3, n + 1):
            new = np.cumsum(new)
            if j <= c:
                new[j:] -= new[:c + 1 - j]
        prob = 2.0 * np.sum(new) / math.factorial(n)
        p_mass_at_c = new[-1] / math.factorial(n)
    else:
        new = np.zeros(c + 1)
        new[0:2] = 1.0
        for j in range(3, n + 1):
            new = np.cumsum(new) / j
            if j <= c:
                new[j:] -= new[:c + 1 - j]
        prob = np.sum(new)
        p_mass_at_c = new[-1] / 2
    if alternative != 'two-sided':
        if in_right_tail == alternative_greater:
            prob /= 2
        else:
            prob = 1 - prob / 2 + p_mass_at_c
    prob = np.clip(prob, 0, 1)
    return prob

def kendalltau(x, y, use_ties=True, use_missing=False, method='auto', alternative='two-sided'):
    if False:
        i = 10
        return i + 15
    '\n    Computes Kendall\'s rank correlation tau on two variables *x* and *y*.\n\n    Parameters\n    ----------\n    x : sequence\n        First data list (for example, time).\n    y : sequence\n        Second data list.\n    use_ties : {True, False}, optional\n        Whether ties correction should be performed.\n    use_missing : {False, True}, optional\n        Whether missing data should be allocated a rank of 0 (False) or the\n        average rank (True)\n    method : {\'auto\', \'asymptotic\', \'exact\'}, optional\n        Defines which method is used to calculate the p-value [1]_.\n        \'asymptotic\' uses a normal approximation valid for large samples.\n        \'exact\' computes the exact p-value, but can only be used if no ties\n        are present. As the sample size increases, the \'exact\' computation\n        time may grow and the result may lose some precision.\n        \'auto\' is the default and selects the appropriate\n        method based on a trade-off between speed and accuracy.\n    alternative : {\'two-sided\', \'less\', \'greater\'}, optional\n        Defines the alternative hypothesis. Default is \'two-sided\'.\n        The following options are available:\n\n        * \'two-sided\': the rank correlation is nonzero\n        * \'less\': the rank correlation is negative (less than zero)\n        * \'greater\':  the rank correlation is positive (greater than zero)\n\n    Returns\n    -------\n    res : SignificanceResult\n        An object containing attributes:\n\n        statistic : float\n           The tau statistic.\n        pvalue : float\n           The p-value for a hypothesis test whose null hypothesis is\n           an absence of association, tau = 0.\n\n    References\n    ----------\n    .. [1] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),\n           Charles Griffin & Co., 1970.\n\n    '
    (x, y, n) = _chk_size(x, y)
    (x, y) = (x.flatten(), y.flatten())
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    if m is not nomask:
        x = ma.array(x, mask=m, copy=True)
        y = ma.array(y, mask=m, copy=True)
        n -= int(m.sum())
    if n < 2:
        res = scipy.stats._stats_py.SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res
    rx = ma.masked_equal(rankdata(x, use_missing=use_missing), 0)
    ry = ma.masked_equal(rankdata(y, use_missing=use_missing), 0)
    idx = rx.argsort()
    (rx, ry) = (rx[idx], ry[idx])
    C = np.sum([((ry[i + 1:] > ry[i]) * (rx[i + 1:] > rx[i])).filled(0).sum() for i in range(len(ry) - 1)], dtype=float)
    D = np.sum([((ry[i + 1:] < ry[i]) * (rx[i + 1:] > rx[i])).filled(0).sum() for i in range(len(ry) - 1)], dtype=float)
    xties = count_tied_groups(x)
    yties = count_tied_groups(y)
    if use_ties:
        corr_x = np.sum([v * k * (k - 1) for (k, v) in xties.items()], dtype=float)
        corr_y = np.sum([v * k * (k - 1) for (k, v) in yties.items()], dtype=float)
        denom = ma.sqrt((n * (n - 1) - corr_x) / 2.0 * (n * (n - 1) - corr_y) / 2.0)
    else:
        denom = n * (n - 1) / 2.0
    tau = (C - D) / denom
    if method == 'exact' and (xties or yties):
        raise ValueError('Ties found, exact method cannot be used.')
    if method == 'auto':
        if (not xties and (not yties)) and (n <= 33 or min(C, n * (n - 1) / 2.0 - C) <= 1):
            method = 'exact'
        else:
            method = 'asymptotic'
    if not xties and (not yties) and (method == 'exact'):
        prob = _kendall_p_exact(n, C, alternative)
    elif method == 'asymptotic':
        var_s = n * (n - 1) * (2 * n + 5)
        if use_ties:
            var_s -= np.sum([v * k * (k - 1) * (2 * k + 5) * 1.0 for (k, v) in xties.items()])
            var_s -= np.sum([v * k * (k - 1) * (2 * k + 5) * 1.0 for (k, v) in yties.items()])
            v1 = np.sum([v * k * (k - 1) for (k, v) in xties.items()], dtype=float) * np.sum([v * k * (k - 1) for (k, v) in yties.items()], dtype=float)
            v1 /= 2.0 * n * (n - 1)
            if n > 2:
                v2 = np.sum([v * k * (k - 1) * (k - 2) for (k, v) in xties.items()], dtype=float) * np.sum([v * k * (k - 1) * (k - 2) for (k, v) in yties.items()], dtype=float)
                v2 /= 9.0 * n * (n - 1) * (n - 2)
            else:
                v2 = 0
        else:
            v1 = v2 = 0
        var_s /= 18.0
        var_s += v1 + v2
        z = (C - D) / np.sqrt(var_s)
        (_, prob) = scipy.stats._stats_py._normtest_finish(z, alternative)
    else:
        raise ValueError('Unknown method ' + str(method) + ' specified, please use auto, exact or asymptotic.')
    res = scipy.stats._stats_py.SignificanceResult(tau, prob)
    res.correlation = tau
    return res

def kendalltau_seasonal(x):
    if False:
        i = 10
        return i + 15
    "\n    Computes a multivariate Kendall's rank correlation tau, for seasonal data.\n\n    Parameters\n    ----------\n    x : 2-D ndarray\n        Array of seasonal data, with seasons in columns.\n\n    "
    x = ma.array(x, subok=True, copy=False, ndmin=2)
    (n, m) = x.shape
    n_p = x.count(0)
    S_szn = sum((msign(x[i:] - x[i]).sum(0) for i in range(n)))
    S_tot = S_szn.sum()
    n_tot = x.count()
    ties = count_tied_groups(x.compressed())
    corr_ties = sum((v * k * (k - 1) for (k, v) in ties.items()))
    denom_tot = ma.sqrt(1.0 * n_tot * (n_tot - 1) * (n_tot * (n_tot - 1) - corr_ties)) / 2.0
    R = rankdata(x, axis=0, use_missing=True)
    K = ma.empty((m, m), dtype=int)
    covmat = ma.empty((m, m), dtype=float)
    denom_szn = ma.empty(m, dtype=float)
    for j in range(m):
        ties_j = count_tied_groups(x[:, j].compressed())
        corr_j = sum((v * k * (k - 1) for (k, v) in ties_j.items()))
        cmb = n_p[j] * (n_p[j] - 1)
        for k in range(j, m, 1):
            K[j, k] = sum((msign((x[i:, j] - x[i, j]) * (x[i:, k] - x[i, k])).sum() for i in range(n)))
            covmat[j, k] = (K[j, k] + 4 * (R[:, j] * R[:, k]).sum() - n * (n_p[j] + 1) * (n_p[k] + 1)) / 3.0
            K[k, j] = K[j, k]
            covmat[k, j] = covmat[j, k]
        denom_szn[j] = ma.sqrt(cmb * (cmb - corr_j)) / 2.0
    var_szn = covmat.diagonal()
    z_szn = msign(S_szn) * (abs(S_szn) - 1) / ma.sqrt(var_szn)
    z_tot_ind = msign(S_tot) * (abs(S_tot) - 1) / ma.sqrt(var_szn.sum())
    z_tot_dep = msign(S_tot) * (abs(S_tot) - 1) / ma.sqrt(covmat.sum())
    prob_szn = special.erfc(abs(z_szn) / np.sqrt(2))
    prob_tot_ind = special.erfc(abs(z_tot_ind) / np.sqrt(2))
    prob_tot_dep = special.erfc(abs(z_tot_dep) / np.sqrt(2))
    chi2_tot = (z_szn * z_szn).sum()
    chi2_trd = m * z_szn.mean() ** 2
    output = {'seasonal tau': S_szn / denom_szn, 'global tau': S_tot / denom_tot, 'global tau (alt)': S_tot / denom_szn.sum(), 'seasonal p-value': prob_szn, 'global p-value (indep)': prob_tot_ind, 'global p-value (dep)': prob_tot_dep, 'chi2 total': chi2_tot, 'chi2 trend': chi2_trd}
    return output
PointbiserialrResult = namedtuple('PointbiserialrResult', ('correlation', 'pvalue'))

def pointbiserialr(x, y):
    if False:
        print('Hello World!')
    'Calculates a point biserial correlation coefficient and its p-value.\n\n    Parameters\n    ----------\n    x : array_like of bools\n        Input array.\n    y : array_like\n        Input array.\n\n    Returns\n    -------\n    correlation : float\n        R value\n    pvalue : float\n        2-tailed p-value\n\n    Notes\n    -----\n    Missing values are considered pair-wise: if a value is missing in x,\n    the corresponding value in y is masked.\n\n    For more details on `pointbiserialr`, see `scipy.stats.pointbiserialr`.\n\n    '
    x = ma.fix_invalid(x, copy=True).astype(bool)
    y = ma.fix_invalid(y, copy=True).astype(float)
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    if m is not nomask:
        unmask = np.logical_not(m)
        x = x[unmask]
        y = y[unmask]
    n = len(x)
    phat = x.sum() / float(n)
    y0 = y[~x]
    y1 = y[x]
    y0m = y0.mean()
    y1m = y1.mean()
    rpb = (y1m - y0m) * np.sqrt(phat * (1 - phat)) / y.std()
    df = n - 2
    t = rpb * ma.sqrt(df / (1.0 - rpb ** 2))
    prob = _betai(0.5 * df, 0.5, df / (df + t * t))
    return PointbiserialrResult(rpb, prob)

def linregress(x, y=None):
    if False:
        while True:
            i = 10
    '\n    Linear regression calculation\n\n    Note that the non-masked version is used, and that this docstring is\n    replaced by the non-masked docstring + some info on missing data.\n\n    '
    if y is None:
        x = ma.array(x)
        if x.shape[0] == 2:
            (x, y) = x
        elif x.shape[1] == 2:
            (x, y) = x.T
        else:
            raise ValueError(f'If only `x` is given as input, it has to be of shape (2, N) or (N, 2), provided shape was {x.shape}')
    else:
        x = ma.array(x)
        y = ma.array(y)
    x = x.flatten()
    y = y.flatten()
    if np.amax(x) == np.amin(x) and len(x) > 1:
        raise ValueError('Cannot calculate a linear regression if all x values are identical')
    m = ma.mask_or(ma.getmask(x), ma.getmask(y), shrink=False)
    if m is not nomask:
        x = ma.array(x, mask=m)
        y = ma.array(y, mask=m)
        if np.any(~m):
            result = stats_linregress(x.data[~m], y.data[~m])
        else:
            result = stats_LinregressResult(slope=None, intercept=None, rvalue=None, pvalue=None, stderr=None, intercept_stderr=None)
    else:
        result = stats_linregress(x.data, y.data)
    return result

def theilslopes(y, x=None, alpha=0.95, method='separate'):
    if False:
        i = 10
        return i + 15
    '\n    Computes the Theil-Sen estimator for a set of points (x, y).\n\n    `theilslopes` implements a method for robust linear regression.  It\n    computes the slope as the median of all slopes between paired values.\n\n    Parameters\n    ----------\n    y : array_like\n        Dependent variable.\n    x : array_like or None, optional\n        Independent variable. If None, use ``arange(len(y))`` instead.\n    alpha : float, optional\n        Confidence degree between 0 and 1. Default is 95% confidence.\n        Note that `alpha` is symmetric around 0.5, i.e. both 0.1 and 0.9 are\n        interpreted as "find the 90% confidence interval".\n    method : {\'joint\', \'separate\'}, optional\n        Method to be used for computing estimate for intercept.\n        Following methods are supported,\n\n            * \'joint\': Uses np.median(y - slope * x) as intercept.\n            * \'separate\': Uses np.median(y) - slope * np.median(x)\n                          as intercept.\n\n        The default is \'separate\'.\n\n        .. versionadded:: 1.8.0\n\n    Returns\n    -------\n    result : ``TheilslopesResult`` instance\n        The return value is an object with the following attributes:\n\n        slope : float\n            Theil slope.\n        intercept : float\n            Intercept of the Theil line.\n        low_slope : float\n            Lower bound of the confidence interval on `slope`.\n        high_slope : float\n            Upper bound of the confidence interval on `slope`.\n\n    See Also\n    --------\n    siegelslopes : a similar technique using repeated medians\n\n\n    Notes\n    -----\n    For more details on `theilslopes`, see `scipy.stats.theilslopes`.\n\n    '
    y = ma.asarray(y).flatten()
    if x is None:
        x = ma.arange(len(y), dtype=float)
    else:
        x = ma.asarray(x).flatten()
        if len(x) != len(y):
            raise ValueError(f'Incompatible lengths ! ({len(y)}<>{len(x)})')
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    y._mask = x._mask = m
    y = y.compressed()
    x = x.compressed().astype(float)
    return stats_theilslopes(y, x, alpha=alpha, method=method)

def siegelslopes(y, x=None, method='hierarchical'):
    if False:
        while True:
            i = 10
    "\n    Computes the Siegel estimator for a set of points (x, y).\n\n    `siegelslopes` implements a method for robust linear regression\n    using repeated medians to fit a line to the points (x, y).\n    The method is robust to outliers with an asymptotic breakdown point\n    of 50%.\n\n    Parameters\n    ----------\n    y : array_like\n        Dependent variable.\n    x : array_like or None, optional\n        Independent variable. If None, use ``arange(len(y))`` instead.\n    method : {'hierarchical', 'separate'}\n        If 'hierarchical', estimate the intercept using the estimated\n        slope ``slope`` (default option).\n        If 'separate', estimate the intercept independent of the estimated\n        slope. See Notes for details.\n\n    Returns\n    -------\n    result : ``SiegelslopesResult`` instance\n        The return value is an object with the following attributes:\n\n        slope : float\n            Estimate of the slope of the regression line.\n        intercept : float\n            Estimate of the intercept of the regression line.\n\n    See Also\n    --------\n    theilslopes : a similar technique without repeated medians\n\n    Notes\n    -----\n    For more details on `siegelslopes`, see `scipy.stats.siegelslopes`.\n\n    "
    y = ma.asarray(y).ravel()
    if x is None:
        x = ma.arange(len(y), dtype=float)
    else:
        x = ma.asarray(x).ravel()
        if len(x) != len(y):
            raise ValueError(f'Incompatible lengths ! ({len(y)}<>{len(x)})')
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    y._mask = x._mask = m
    y = y.compressed()
    x = x.compressed().astype(float)
    return stats_siegelslopes(y, x, method=method)
SenSeasonalSlopesResult = _make_tuple_bunch('SenSeasonalSlopesResult', ['intra_slope', 'inter_slope'])

def sen_seasonal_slopes(x):
    if False:
        print('Hello World!')
    '\n    Computes seasonal Theil-Sen and Kendall slope estimators.\n\n    The seasonal generalization of Sen\'s slope computes the slopes between all\n    pairs of values within a "season" (column) of a 2D array. It returns an\n    array containing the median of these "within-season" slopes for each\n    season (the Theil-Sen slope estimator of each season), and it returns the\n    median of the within-season slopes across all seasons (the seasonal Kendall\n    slope estimator).\n\n    Parameters\n    ----------\n    x : 2D array_like\n        Each column of `x` contains measurements of the dependent variable\n        within a season. The independent variable (usually time) of each season\n        is assumed to be ``np.arange(x.shape[0])``.\n\n    Returns\n    -------\n    result : ``SenSeasonalSlopesResult`` instance\n        The return value is an object with the following attributes:\n\n        intra_slope : ndarray\n            For each season, the Theil-Sen slope estimator: the median of\n            within-season slopes.\n        inter_slope : float\n            The seasonal Kendall slope estimateor: the median of within-season\n            slopes *across all* seasons.\n\n    See Also\n    --------\n    theilslopes : the analogous function for non-seasonal data\n    scipy.stats.theilslopes : non-seasonal slopes for non-masked arrays\n\n    Notes\n    -----\n    The slopes :math:`d_{ijk}` within season :math:`i` are:\n\n    .. math::\n\n        d_{ijk} = \\frac{x_{ij} - x_{ik}}\n                            {j - k}\n\n    for pairs of distinct integer indices :math:`j, k` of :math:`x`.\n\n    Element :math:`i` of the returned `intra_slope` array is the median of the\n    :math:`d_{ijk}` over all :math:`j < k`; this is the Theil-Sen slope\n    estimator of season :math:`i`. The returned `inter_slope` value, better\n    known as the seasonal Kendall slope estimator, is the median of the\n    :math:`d_{ijk}` over all :math:`i, j, k`.\n\n    References\n    ----------\n    .. [1] Hirsch, Robert M., James R. Slack, and Richard A. Smith.\n           "Techniques of trend analysis for monthly water quality data."\n           *Water Resources Research* 18.1 (1982): 107-121.\n\n    Examples\n    --------\n    Suppose we have 100 observations of a dependent variable for each of four\n    seasons:\n\n    >>> import numpy as np\n    >>> rng = np.random.default_rng()\n    >>> x = rng.random(size=(100, 4))\n\n    We compute the seasonal slopes as:\n\n    >>> from scipy import stats\n    >>> intra_slope, inter_slope = stats.mstats.sen_seasonal_slopes(x)\n\n    If we define a function to compute all slopes between observations within\n    a season:\n\n    >>> def dijk(yi):\n    ...     n = len(yi)\n    ...     x = np.arange(n)\n    ...     dy = yi - yi[:, np.newaxis]\n    ...     dx = x - x[:, np.newaxis]\n    ...     # we only want unique pairs of distinct indices\n    ...     mask = np.triu(np.ones((n, n), dtype=bool), k=1)\n    ...     return dy[mask]/dx[mask]\n\n    then element ``i`` of ``intra_slope`` is the median of ``dijk[x[:, i]]``:\n\n    >>> i = 2\n    >>> np.allclose(np.median(dijk(x[:, i])), intra_slope[i])\n    True\n\n    and ``inter_slope`` is the median of the values returned by ``dijk`` for\n    all seasons:\n\n    >>> all_slopes = np.concatenate([dijk(x[:, i]) for i in range(x.shape[1])])\n    >>> np.allclose(np.median(all_slopes), inter_slope)\n    True\n\n    Because the data are randomly generated, we would expect the median slopes\n    to be nearly zero both within and across all seasons, and indeed they are:\n\n    >>> intra_slope.data\n    array([ 0.00124504, -0.00277761, -0.00221245, -0.00036338])\n    >>> inter_slope\n    -0.0010511779872922058\n\n    '
    x = ma.array(x, subok=True, copy=False, ndmin=2)
    (n, _) = x.shape
    szn_slopes = ma.vstack([(x[i + 1:] - x[i]) / np.arange(1, n - i)[:, None] for i in range(n)])
    szn_medslopes = ma.median(szn_slopes, axis=0)
    medslope = ma.median(szn_slopes, axis=None)
    return SenSeasonalSlopesResult(szn_medslopes, medslope)
Ttest_1sampResult = namedtuple('Ttest_1sampResult', ('statistic', 'pvalue'))

def ttest_1samp(a, popmean, axis=0, alternative='two-sided'):
    if False:
        return 10
    "\n    Calculates the T-test for the mean of ONE group of scores.\n\n    Parameters\n    ----------\n    a : array_like\n        sample observation\n    popmean : float or array_like\n        expected value in null hypothesis, if array_like than it must have the\n        same shape as `a` excluding the axis dimension\n    axis : int or None, optional\n        Axis along which to compute test. If None, compute over the whole\n        array `a`.\n    alternative : {'two-sided', 'less', 'greater'}, optional\n        Defines the alternative hypothesis.\n        The following options are available (default is 'two-sided'):\n\n        * 'two-sided': the mean of the underlying distribution of the sample\n          is different than the given population mean (`popmean`)\n        * 'less': the mean of the underlying distribution of the sample is\n          less than the given population mean (`popmean`)\n        * 'greater': the mean of the underlying distribution of the sample is\n          greater than the given population mean (`popmean`)\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    statistic : float or array\n        t-statistic\n    pvalue : float or array\n        The p-value\n\n    Notes\n    -----\n    For more details on `ttest_1samp`, see `scipy.stats.ttest_1samp`.\n\n    "
    (a, axis) = _chk_asarray(a, axis)
    if a.size == 0:
        return (np.nan, np.nan)
    x = a.mean(axis=axis)
    v = a.var(axis=axis, ddof=1)
    n = a.count(axis=axis)
    df = ma.asanyarray(n - 1.0)
    svar = (n - 1.0) * v / df
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x - popmean) / ma.sqrt(svar / n)
    (t, prob) = scipy.stats._stats_py._ttest_finish(df, t, alternative)
    return Ttest_1sampResult(t, prob)
ttest_onesamp = ttest_1samp
Ttest_indResult = namedtuple('Ttest_indResult', ('statistic', 'pvalue'))

def ttest_ind(a, b, axis=0, equal_var=True, alternative='two-sided'):
    if False:
        i = 10
        return i + 15
    "\n    Calculates the T-test for the means of TWO INDEPENDENT samples of scores.\n\n    Parameters\n    ----------\n    a, b : array_like\n        The arrays must have the same shape, except in the dimension\n        corresponding to `axis` (the first, by default).\n    axis : int or None, optional\n        Axis along which to compute test. If None, compute over the whole\n        arrays, `a`, and `b`.\n    equal_var : bool, optional\n        If True, perform a standard independent 2 sample test that assumes equal\n        population variances.\n        If False, perform Welch's t-test, which does not assume equal population\n        variance.\n\n        .. versionadded:: 0.17.0\n    alternative : {'two-sided', 'less', 'greater'}, optional\n        Defines the alternative hypothesis.\n        The following options are available (default is 'two-sided'):\n\n        * 'two-sided': the means of the distributions underlying the samples\n          are unequal.\n        * 'less': the mean of the distribution underlying the first sample\n          is less than the mean of the distribution underlying the second\n          sample.\n        * 'greater': the mean of the distribution underlying the first\n          sample is greater than the mean of the distribution underlying\n          the second sample.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    statistic : float or array\n        The calculated t-statistic.\n    pvalue : float or array\n        The p-value.\n\n    Notes\n    -----\n    For more details on `ttest_ind`, see `scipy.stats.ttest_ind`.\n\n    "
    (a, b, axis) = _chk2_asarray(a, b, axis)
    if a.size == 0 or b.size == 0:
        return Ttest_indResult(np.nan, np.nan)
    (x1, x2) = (a.mean(axis), b.mean(axis))
    (v1, v2) = (a.var(axis=axis, ddof=1), b.var(axis=axis, ddof=1))
    (n1, n2) = (a.count(axis), b.count(axis))
    if equal_var:
        df = ma.asanyarray(n1 + n2 - 2.0)
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        denom = ma.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))
        df = np.where(np.isnan(df), 1, df)
        denom = ma.sqrt(vn1 + vn2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x1 - x2) / denom
    (t, prob) = scipy.stats._stats_py._ttest_finish(df, t, alternative)
    return Ttest_indResult(t, prob)
Ttest_relResult = namedtuple('Ttest_relResult', ('statistic', 'pvalue'))

def ttest_rel(a, b, axis=0, alternative='two-sided'):
    if False:
        i = 10
        return i + 15
    "\n    Calculates the T-test on TWO RELATED samples of scores, a and b.\n\n    Parameters\n    ----------\n    a, b : array_like\n        The arrays must have the same shape.\n    axis : int or None, optional\n        Axis along which to compute test. If None, compute over the whole\n        arrays, `a`, and `b`.\n    alternative : {'two-sided', 'less', 'greater'}, optional\n        Defines the alternative hypothesis.\n        The following options are available (default is 'two-sided'):\n\n        * 'two-sided': the means of the distributions underlying the samples\n          are unequal.\n        * 'less': the mean of the distribution underlying the first sample\n          is less than the mean of the distribution underlying the second\n          sample.\n        * 'greater': the mean of the distribution underlying the first\n          sample is greater than the mean of the distribution underlying\n          the second sample.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    statistic : float or array\n        t-statistic\n    pvalue : float or array\n        two-tailed p-value\n\n    Notes\n    -----\n    For more details on `ttest_rel`, see `scipy.stats.ttest_rel`.\n\n    "
    (a, b, axis) = _chk2_asarray(a, b, axis)
    if len(a) != len(b):
        raise ValueError('unequal length arrays')
    if a.size == 0 or b.size == 0:
        return Ttest_relResult(np.nan, np.nan)
    n = a.count(axis)
    df = ma.asanyarray(n - 1.0)
    d = (a - b).astype('d')
    dm = d.mean(axis)
    v = d.var(axis=axis, ddof=1)
    denom = ma.sqrt(v / n)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = dm / denom
    (t, prob) = scipy.stats._stats_py._ttest_finish(df, t, alternative)
    return Ttest_relResult(t, prob)
MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))

def mannwhitneyu(x, y, use_continuity=True):
    if False:
        i = 10
        return i + 15
    '\n    Computes the Mann-Whitney statistic\n\n    Missing values in `x` and/or `y` are discarded.\n\n    Parameters\n    ----------\n    x : sequence\n        Input\n    y : sequence\n        Input\n    use_continuity : {True, False}, optional\n        Whether a continuity correction (1/2.) should be taken into account.\n\n    Returns\n    -------\n    statistic : float\n        The minimum of the Mann-Whitney statistics\n    pvalue : float\n        Approximate two-sided p-value assuming a normal distribution.\n\n    '
    x = ma.asarray(x).compressed().view(ndarray)
    y = ma.asarray(y).compressed().view(ndarray)
    ranks = rankdata(np.concatenate([x, y]))
    (nx, ny) = (len(x), len(y))
    nt = nx + ny
    U = ranks[:nx].sum() - nx * (nx + 1) / 2.0
    U = max(U, nx * ny - U)
    u = nx * ny - U
    mu = nx * ny / 2.0
    sigsq = (nt ** 3 - nt) / 12.0
    ties = count_tied_groups(ranks)
    sigsq -= sum((v * (k ** 3 - k) for (k, v) in ties.items())) / 12.0
    sigsq *= nx * ny / float(nt * (nt - 1))
    if use_continuity:
        z = (U - 1 / 2.0 - mu) / ma.sqrt(sigsq)
    else:
        z = (U - mu) / ma.sqrt(sigsq)
    prob = special.erfc(abs(z) / np.sqrt(2))
    return MannwhitneyuResult(u, prob)
KruskalResult = namedtuple('KruskalResult', ('statistic', 'pvalue'))

def kruskal(*args):
    if False:
        while True:
            i = 10
    "\n    Compute the Kruskal-Wallis H-test for independent samples\n\n    Parameters\n    ----------\n    sample1, sample2, ... : array_like\n       Two or more arrays with the sample measurements can be given as\n       arguments.\n\n    Returns\n    -------\n    statistic : float\n       The Kruskal-Wallis H statistic, corrected for ties\n    pvalue : float\n       The p-value for the test using the assumption that H has a chi\n       square distribution\n\n    Notes\n    -----\n    For more details on `kruskal`, see `scipy.stats.kruskal`.\n\n    Examples\n    --------\n    >>> from scipy.stats.mstats import kruskal\n\n    Random samples from three different brands of batteries were tested\n    to see how long the charge lasted. Results were as follows:\n\n    >>> a = [6.3, 5.4, 5.7, 5.2, 5.0]\n    >>> b = [6.9, 7.0, 6.1, 7.9]\n    >>> c = [7.2, 6.9, 6.1, 6.5]\n\n    Test the hypothesis that the distribution functions for all of the brands'\n    durations are identical. Use 5% level of significance.\n\n    >>> kruskal(a, b, c)\n    KruskalResult(statistic=7.113812154696133, pvalue=0.028526948491942164)\n\n    The null hypothesis is rejected at the 5% level of significance\n    because the returned p-value is less than the critical value of 5%.\n\n    "
    output = argstoarray(*args)
    ranks = ma.masked_equal(rankdata(output, use_missing=False), 0)
    sumrk = ranks.sum(-1)
    ngrp = ranks.count(-1)
    ntot = ranks.count()
    H = 12.0 / (ntot * (ntot + 1)) * (sumrk ** 2 / ngrp).sum() - 3 * (ntot + 1)
    ties = count_tied_groups(ranks)
    T = 1.0 - sum((v * (k ** 3 - k) for (k, v) in ties.items())) / float(ntot ** 3 - ntot)
    if T == 0:
        raise ValueError('All numbers are identical in kruskal')
    H /= T
    df = len(output) - 1
    prob = distributions.chi2.sf(H, df)
    return KruskalResult(H, prob)
kruskalwallis = kruskal

@_rename_parameter('mode', 'method')
def ks_1samp(x, cdf, args=(), alternative='two-sided', method='auto'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Computes the Kolmogorov-Smirnov test on one sample of masked values.\n\n    Missing values in `x` are discarded.\n\n    Parameters\n    ----------\n    x : array_like\n        a 1-D array of observations of random variables.\n    cdf : str or callable\n        If a string, it should be the name of a distribution in `scipy.stats`.\n        If a callable, that callable is used to calculate the cdf.\n    args : tuple, sequence, optional\n        Distribution parameters, used if `cdf` is a string.\n    alternative : {'two-sided', 'less', 'greater'}, optional\n        Indicates the alternative hypothesis.  Default is 'two-sided'.\n    method : {'auto', 'exact', 'asymp'}, optional\n        Defines the method used for calculating the p-value.\n        The following options are available (default is 'auto'):\n\n          * 'auto' : use 'exact' for small size arrays, 'asymp' for large\n          * 'exact' : use approximation to exact distribution of test statistic\n          * 'asymp' : use asymptotic distribution of test statistic\n\n    Returns\n    -------\n    d : float\n        Value of the Kolmogorov Smirnov test\n    p : float\n        Corresponding p-value.\n\n    "
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(alternative.lower()[0], alternative)
    return scipy.stats._stats_py.ks_1samp(x, cdf, args=args, alternative=alternative, method=method)

@_rename_parameter('mode', 'method')
def ks_2samp(data1, data2, alternative='two-sided', method='auto'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Computes the Kolmogorov-Smirnov test on two samples.\n\n    Missing values in `x` and/or `y` are discarded.\n\n    Parameters\n    ----------\n    data1 : array_like\n        First data set\n    data2 : array_like\n        Second data set\n    alternative : {'two-sided', 'less', 'greater'}, optional\n        Indicates the alternative hypothesis.  Default is 'two-sided'.\n    method : {'auto', 'exact', 'asymp'}, optional\n        Defines the method used for calculating the p-value.\n        The following options are available (default is 'auto'):\n\n          * 'auto' : use 'exact' for small size arrays, 'asymp' for large\n          * 'exact' : use approximation to exact distribution of test statistic\n          * 'asymp' : use asymptotic distribution of test statistic\n\n    Returns\n    -------\n    d : float\n        Value of the Kolmogorov Smirnov test\n    p : float\n        Corresponding p-value.\n\n    "
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(alternative.lower()[0], alternative)
    return scipy.stats._stats_py.ks_2samp(data1, data2, alternative=alternative, method=method)
ks_twosamp = ks_2samp

@_rename_parameter('mode', 'method')
def kstest(data1, data2, args=(), alternative='two-sided', method='auto'):
    if False:
        return 10
    '\n\n    Parameters\n    ----------\n    data1 : array_like\n    data2 : str, callable or array_like\n    args : tuple, sequence, optional\n        Distribution parameters, used if `data1` or `data2` are strings.\n    alternative : str, as documented in stats.kstest\n    method : str, as documented in stats.kstest\n\n    Returns\n    -------\n    tuple of (K-S statistic, probability)\n\n    '
    return scipy.stats._stats_py.kstest(data1, data2, args, alternative=alternative, method=method)

def trima(a, limits=None, inclusive=(True, True)):
    if False:
        for i in range(10):
            print('nop')
    '\n    Trims an array by masking the data outside some given limits.\n\n    Returns a masked version of the input array.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    limits : {None, tuple}, optional\n        Tuple of (lower limit, upper limit) in absolute values.\n        Values of the input array lower (greater) than the lower (upper) limit\n        will be masked.  A limit is None indicates an open interval.\n    inclusive : (bool, bool) tuple, optional\n        Tuple of (lower flag, upper flag), indicating whether values exactly\n        equal to the lower (upper) limit are allowed.\n\n    Examples\n    --------\n    >>> from scipy.stats.mstats import trima\n    >>> import numpy as np\n\n    >>> a = np.arange(10)\n\n    The interval is left-closed and right-open, i.e., `[2, 8)`.\n    Trim the array by keeping only values in the interval.\n\n    >>> trima(a, limits=(2, 8), inclusive=(True, False))\n    masked_array(data=[--, --, 2, 3, 4, 5, 6, 7, --, --],\n                 mask=[ True,  True, False, False, False, False, False, False,\n                        True,  True],\n           fill_value=999999)\n\n    '
    a = ma.asarray(a)
    a.unshare_mask()
    if limits is None or limits == (None, None):
        return a
    (lower_lim, upper_lim) = limits
    (lower_in, upper_in) = inclusive
    condition = False
    if lower_lim is not None:
        if lower_in:
            condition |= a < lower_lim
        else:
            condition |= a <= lower_lim
    if upper_lim is not None:
        if upper_in:
            condition |= a > upper_lim
        else:
            condition |= a >= upper_lim
    a[condition.filled(True)] = masked
    return a

def trimr(a, limits=None, inclusive=(True, True), axis=None):
    if False:
        while True:
            i = 10
    '\n    Trims an array by masking some proportion of the data on each end.\n    Returns a masked version of the input array.\n\n    Parameters\n    ----------\n    a : sequence\n        Input array.\n    limits : {None, tuple}, optional\n        Tuple of the percentages to cut on each side of the array, with respect\n        to the number of unmasked data, as floats between 0. and 1.\n        Noting n the number of unmasked data before trimming, the\n        (n*limits[0])th smallest data and the (n*limits[1])th largest data are\n        masked, and the total number of unmasked data after trimming is\n        n*(1.-sum(limits)).  The value of one limit can be set to None to\n        indicate an open interval.\n    inclusive : {(True,True) tuple}, optional\n        Tuple of flags indicating whether the number of data being masked on\n        the left (right) end should be truncated (True) or rounded (False) to\n        integers.\n    axis : {None,int}, optional\n        Axis along which to trim. If None, the whole array is trimmed, but its\n        shape is maintained.\n\n    '

    def _trimr1D(a, low_limit, up_limit, low_inclusive, up_inclusive):
        if False:
            i = 10
            return i + 15
        n = a.count()
        idx = a.argsort()
        if low_limit:
            if low_inclusive:
                lowidx = int(low_limit * n)
            else:
                lowidx = int(np.round(low_limit * n))
            a[idx[:lowidx]] = masked
        if up_limit is not None:
            if up_inclusive:
                upidx = n - int(n * up_limit)
            else:
                upidx = n - int(np.round(n * up_limit))
            a[idx[upidx:]] = masked
        return a
    a = ma.asarray(a)
    a.unshare_mask()
    if limits is None:
        return a
    (lolim, uplim) = limits
    errmsg = 'The proportion to cut from the %s should be between 0. and 1.'
    if lolim is not None:
        if lolim > 1.0 or lolim < 0:
            raise ValueError(errmsg % 'beginning' + '(got %s)' % lolim)
    if uplim is not None:
        if uplim > 1.0 or uplim < 0:
            raise ValueError(errmsg % 'end' + '(got %s)' % uplim)
    (loinc, upinc) = inclusive
    if axis is None:
        shp = a.shape
        return _trimr1D(a.ravel(), lolim, uplim, loinc, upinc).reshape(shp)
    else:
        return ma.apply_along_axis(_trimr1D, axis, a, lolim, uplim, loinc, upinc)
trimdoc = '\n    Parameters\n    ----------\n    a : sequence\n        Input array\n    limits : {None, tuple}, optional\n        If `relative` is False, tuple (lower limit, upper limit) in absolute values.\n        Values of the input array lower (greater) than the lower (upper) limit are\n        masked.\n\n        If `relative` is True, tuple (lower percentage, upper percentage) to cut\n        on each side of the  array, with respect to the number of unmasked data.\n\n        Noting n the number of unmasked data before trimming, the (n*limits[0])th\n        smallest data and the (n*limits[1])th largest data are masked, and the\n        total number of unmasked data after trimming is n*(1.-sum(limits))\n        In each case, the value of one limit can be set to None to indicate an\n        open interval.\n\n        If limits is None, no trimming is performed\n    inclusive : {(bool, bool) tuple}, optional\n        If `relative` is False, tuple indicating whether values exactly equal\n        to the absolute limits are allowed.\n        If `relative` is True, tuple indicating whether the number of data\n        being masked on each side should be rounded (True) or truncated\n        (False).\n    relative : bool, optional\n        Whether to consider the limits as absolute values (False) or proportions\n        to cut (True).\n    axis : int, optional\n        Axis along which to trim.\n'

def trim(a, limits=None, inclusive=(True, True), relative=False, axis=None):
    if False:
        print('Hello World!')
    '\n    Trims an array by masking the data outside some given limits.\n\n    Returns a masked version of the input array.\n\n    %s\n\n    Examples\n    --------\n    >>> from scipy.stats.mstats import trim\n    >>> z = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10]\n    >>> print(trim(z,(3,8)))\n    [-- -- 3 4 5 6 7 8 -- --]\n    >>> print(trim(z,(0.1,0.2),relative=True))\n    [-- 2 3 4 5 6 7 8 -- --]\n\n    '
    if relative:
        return trimr(a, limits=limits, inclusive=inclusive, axis=axis)
    else:
        return trima(a, limits=limits, inclusive=inclusive)
if trim.__doc__:
    trim.__doc__ = trim.__doc__ % trimdoc

def trimboth(data, proportiontocut=0.2, inclusive=(True, True), axis=None):
    if False:
        return 10
    '\n    Trims the smallest and largest data values.\n\n    Trims the `data` by masking the ``int(proportiontocut * n)`` smallest and\n    ``int(proportiontocut * n)`` largest values of data along the given axis,\n    where n is the number of unmasked values before trimming.\n\n    Parameters\n    ----------\n    data : ndarray\n        Data to trim.\n    proportiontocut : float, optional\n        Percentage of trimming (as a float between 0 and 1).\n        If n is the number of unmasked values before trimming, the number of\n        values after trimming is ``(1 - 2*proportiontocut) * n``.\n        Default is 0.2.\n    inclusive : {(bool, bool) tuple}, optional\n        Tuple indicating whether the number of data being masked on each side\n        should be rounded (True) or truncated (False).\n    axis : int, optional\n        Axis along which to perform the trimming.\n        If None, the input array is first flattened.\n\n    '
    return trimr(data, limits=(proportiontocut, proportiontocut), inclusive=inclusive, axis=axis)

def trimtail(data, proportiontocut=0.2, tail='left', inclusive=(True, True), axis=None):
    if False:
        i = 10
        return i + 15
    "\n    Trims the data by masking values from one tail.\n\n    Parameters\n    ----------\n    data : array_like\n        Data to trim.\n    proportiontocut : float, optional\n        Percentage of trimming. If n is the number of unmasked values\n        before trimming, the number of values after trimming is\n        ``(1 - proportiontocut) * n``.  Default is 0.2.\n    tail : {'left','right'}, optional\n        If 'left' the `proportiontocut` lowest values will be masked.\n        If 'right' the `proportiontocut` highest values will be masked.\n        Default is 'left'.\n    inclusive : {(bool, bool) tuple}, optional\n        Tuple indicating whether the number of data being masked on each side\n        should be rounded (True) or truncated (False).  Default is\n        (True, True).\n    axis : int, optional\n        Axis along which to perform the trimming.\n        If None, the input array is first flattened.  Default is None.\n\n    Returns\n    -------\n    trimtail : ndarray\n        Returned array of same shape as `data` with masked tail values.\n\n    "
    tail = str(tail).lower()[0]
    if tail == 'l':
        limits = (proportiontocut, None)
    elif tail == 'r':
        limits = (None, proportiontocut)
    else:
        raise TypeError("The tail argument should be in ('left','right')")
    return trimr(data, limits=limits, axis=axis, inclusive=inclusive)
trim1 = trimtail

def trimmed_mean(a, limits=(0.1, 0.1), inclusive=(1, 1), relative=True, axis=None):
    if False:
        return 10
    'Returns the trimmed mean of the data along the given axis.\n\n    %s\n\n    '
    if not isinstance(limits, tuple) and isinstance(limits, float):
        limits = (limits, limits)
    if relative:
        return trimr(a, limits=limits, inclusive=inclusive, axis=axis).mean(axis=axis)
    else:
        return trima(a, limits=limits, inclusive=inclusive).mean(axis=axis)
if trimmed_mean.__doc__:
    trimmed_mean.__doc__ = trimmed_mean.__doc__ % trimdoc

def trimmed_var(a, limits=(0.1, 0.1), inclusive=(1, 1), relative=True, axis=None, ddof=0):
    if False:
        return 10
    'Returns the trimmed variance of the data along the given axis.\n\n    %s\n    ddof : {0,integer}, optional\n        Means Delta Degrees of Freedom. The denominator used during computations\n        is (n-ddof). DDOF=0 corresponds to a biased estimate, DDOF=1 to an un-\n        biased estimate of the variance.\n\n    '
    if not isinstance(limits, tuple) and isinstance(limits, float):
        limits = (limits, limits)
    if relative:
        out = trimr(a, limits=limits, inclusive=inclusive, axis=axis)
    else:
        out = trima(a, limits=limits, inclusive=inclusive)
    return out.var(axis=axis, ddof=ddof)
if trimmed_var.__doc__:
    trimmed_var.__doc__ = trimmed_var.__doc__ % trimdoc

def trimmed_std(a, limits=(0.1, 0.1), inclusive=(1, 1), relative=True, axis=None, ddof=0):
    if False:
        for i in range(10):
            print('nop')
    'Returns the trimmed standard deviation of the data along the given axis.\n\n    %s\n    ddof : {0,integer}, optional\n        Means Delta Degrees of Freedom. The denominator used during computations\n        is (n-ddof). DDOF=0 corresponds to a biased estimate, DDOF=1 to an un-\n        biased estimate of the variance.\n\n    '
    if not isinstance(limits, tuple) and isinstance(limits, float):
        limits = (limits, limits)
    if relative:
        out = trimr(a, limits=limits, inclusive=inclusive, axis=axis)
    else:
        out = trima(a, limits=limits, inclusive=inclusive)
    return out.std(axis=axis, ddof=ddof)
if trimmed_std.__doc__:
    trimmed_std.__doc__ = trimmed_std.__doc__ % trimdoc

def trimmed_stde(a, limits=(0.1, 0.1), inclusive=(1, 1), axis=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns the standard error of the trimmed mean along the given axis.\n\n    Parameters\n    ----------\n    a : sequence\n        Input array\n    limits : {(0.1,0.1), tuple of float}, optional\n        tuple (lower percentage, upper percentage) to cut  on each side of the\n        array, with respect to the number of unmasked data.\n\n        If n is the number of unmasked data before trimming, the values\n        smaller than ``n * limits[0]`` and the values larger than\n        ``n * `limits[1]`` are masked, and the total number of unmasked\n        data after trimming is ``n * (1.-sum(limits))``.  In each case,\n        the value of one limit can be set to None to indicate an open interval.\n        If `limits` is None, no trimming is performed.\n    inclusive : {(bool, bool) tuple} optional\n        Tuple indicating whether the number of data being masked on each side\n        should be rounded (True) or truncated (False).\n    axis : int, optional\n        Axis along which to trim.\n\n    Returns\n    -------\n    trimmed_stde : scalar or ndarray\n\n    '

    def _trimmed_stde_1D(a, low_limit, up_limit, low_inclusive, up_inclusive):
        if False:
            for i in range(10):
                print('nop')
        'Returns the standard error of the trimmed mean for a 1D input data.'
        n = a.count()
        idx = a.argsort()
        if low_limit:
            if low_inclusive:
                lowidx = int(low_limit * n)
            else:
                lowidx = np.round(low_limit * n)
            a[idx[:lowidx]] = masked
        if up_limit is not None:
            if up_inclusive:
                upidx = n - int(n * up_limit)
            else:
                upidx = n - np.round(n * up_limit)
            a[idx[upidx:]] = masked
        a[idx[:lowidx]] = a[idx[lowidx]]
        a[idx[upidx:]] = a[idx[upidx - 1]]
        winstd = a.std(ddof=1)
        return winstd / ((1 - low_limit - up_limit) * np.sqrt(len(a)))
    a = ma.array(a, copy=True, subok=True)
    a.unshare_mask()
    if limits is None:
        return a.std(axis=axis, ddof=1) / ma.sqrt(a.count(axis))
    if not isinstance(limits, tuple) and isinstance(limits, float):
        limits = (limits, limits)
    (lolim, uplim) = limits
    errmsg = 'The proportion to cut from the %s should be between 0. and 1.'
    if lolim is not None:
        if lolim > 1.0 or lolim < 0:
            raise ValueError(errmsg % 'beginning' + '(got %s)' % lolim)
    if uplim is not None:
        if uplim > 1.0 or uplim < 0:
            raise ValueError(errmsg % 'end' + '(got %s)' % uplim)
    (loinc, upinc) = inclusive
    if axis is None:
        return _trimmed_stde_1D(a.ravel(), lolim, uplim, loinc, upinc)
    else:
        if a.ndim > 2:
            raise ValueError("Array 'a' must be at most two dimensional, but got a.ndim = %d" % a.ndim)
        return ma.apply_along_axis(_trimmed_stde_1D, axis, a, lolim, uplim, loinc, upinc)

def _mask_to_limits(a, limits, inclusive):
    if False:
        i = 10
        return i + 15
    'Mask an array for values outside of given limits.\n\n    This is primarily a utility function.\n\n    Parameters\n    ----------\n    a : array\n    limits : (float or None, float or None)\n    A tuple consisting of the (lower limit, upper limit).  Values in the\n    input array less than the lower limit or greater than the upper limit\n    will be masked out. None implies no limit.\n    inclusive : (bool, bool)\n    A tuple consisting of the (lower flag, upper flag).  These flags\n    determine whether values exactly equal to lower or upper are allowed.\n\n    Returns\n    -------\n    A MaskedArray.\n\n    Raises\n    ------\n    A ValueError if there are no values within the given limits.\n    '
    (lower_limit, upper_limit) = limits
    (lower_include, upper_include) = inclusive
    am = ma.MaskedArray(a)
    if lower_limit is not None:
        if lower_include:
            am = ma.masked_less(am, lower_limit)
        else:
            am = ma.masked_less_equal(am, lower_limit)
    if upper_limit is not None:
        if upper_include:
            am = ma.masked_greater(am, upper_limit)
        else:
            am = ma.masked_greater_equal(am, upper_limit)
    if am.count() == 0:
        raise ValueError('No array values within given limits')
    return am

def tmean(a, limits=None, inclusive=(True, True), axis=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the trimmed mean.\n\n    Parameters\n    ----------\n    a : array_like\n        Array of values.\n    limits : None or (lower limit, upper limit), optional\n        Values in the input array less than the lower limit or greater than the\n        upper limit will be ignored.  When limits is None (default), then all\n        values are used.  Either of the limit values in the tuple can also be\n        None representing a half-open interval.\n    inclusive : (bool, bool), optional\n        A tuple consisting of the (lower flag, upper flag).  These flags\n        determine whether values exactly equal to the lower or upper limits\n        are included.  The default value is (True, True).\n    axis : int or None, optional\n        Axis along which to operate. If None, compute over the\n        whole array. Default is None.\n\n    Returns\n    -------\n    tmean : float\n\n    Notes\n    -----\n    For more details on `tmean`, see `scipy.stats.tmean`.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats import mstats\n    >>> a = np.array([[6, 8, 3, 0],\n    ...               [3, 9, 1, 2],\n    ...               [8, 7, 8, 2],\n    ...               [5, 6, 0, 2],\n    ...               [4, 5, 5, 2]])\n    ...\n    ...\n    >>> mstats.tmean(a, (2,5))\n    3.3\n    >>> mstats.tmean(a, (2,5), axis=0)\n    masked_array(data=[4.0, 5.0, 4.0, 2.0],\n                 mask=[False, False, False, False],\n           fill_value=1e+20)\n\n    '
    return trima(a, limits=limits, inclusive=inclusive).mean(axis=axis)

def tvar(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    if False:
        i = 10
        return i + 15
    '\n    Compute the trimmed variance\n\n    This function computes the sample variance of an array of values,\n    while ignoring values which are outside of given `limits`.\n\n    Parameters\n    ----------\n    a : array_like\n        Array of values.\n    limits : None or (lower limit, upper limit), optional\n        Values in the input array less than the lower limit or greater than the\n        upper limit will be ignored. When limits is None, then all values are\n        used. Either of the limit values in the tuple can also be None\n        representing a half-open interval.  The default value is None.\n    inclusive : (bool, bool), optional\n        A tuple consisting of the (lower flag, upper flag).  These flags\n        determine whether values exactly equal to the lower or upper limits\n        are included.  The default value is (True, True).\n    axis : int or None, optional\n        Axis along which to operate. If None, compute over the\n        whole array. Default is zero.\n    ddof : int, optional\n        Delta degrees of freedom. Default is 1.\n\n    Returns\n    -------\n    tvar : float\n        Trimmed variance.\n\n    Notes\n    -----\n    For more details on `tvar`, see `scipy.stats.tvar`.\n\n    '
    a = a.astype(float).ravel()
    if limits is None:
        n = (~a.mask).sum()
        return np.ma.var(a) * n / (n - 1.0)
    am = _mask_to_limits(a, limits=limits, inclusive=inclusive)
    return np.ma.var(am, axis=axis, ddof=ddof)

def tmin(a, lowerlimit=None, axis=0, inclusive=True):
    if False:
        print('Hello World!')
    '\n    Compute the trimmed minimum\n\n    Parameters\n    ----------\n    a : array_like\n        array of values\n    lowerlimit : None or float, optional\n        Values in the input array less than the given limit will be ignored.\n        When lowerlimit is None, then all values are used. The default value\n        is None.\n    axis : int or None, optional\n        Axis along which to operate. Default is 0. If None, compute over the\n        whole array `a`.\n    inclusive : {True, False}, optional\n        This flag determines whether values exactly equal to the lower limit\n        are included.  The default value is True.\n\n    Returns\n    -------\n    tmin : float, int or ndarray\n\n    Notes\n    -----\n    For more details on `tmin`, see `scipy.stats.tmin`.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats import mstats\n    >>> a = np.array([[6, 8, 3, 0],\n    ...               [3, 2, 1, 2],\n    ...               [8, 1, 8, 2],\n    ...               [5, 3, 0, 2],\n    ...               [4, 7, 5, 2]])\n    ...\n    >>> mstats.tmin(a, 5)\n    masked_array(data=[5, 7, 5, --],\n                 mask=[False, False, False,  True],\n           fill_value=999999)\n\n    '
    (a, axis) = _chk_asarray(a, axis)
    am = trima(a, (lowerlimit, None), (inclusive, False))
    return ma.minimum.reduce(am, axis)

def tmax(a, upperlimit=None, axis=0, inclusive=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the trimmed maximum\n\n    This function computes the maximum value of an array along a given axis,\n    while ignoring values larger than a specified upper limit.\n\n    Parameters\n    ----------\n    a : array_like\n        array of values\n    upperlimit : None or float, optional\n        Values in the input array greater than the given limit will be ignored.\n        When upperlimit is None, then all values are used. The default value\n        is None.\n    axis : int or None, optional\n        Axis along which to operate. Default is 0. If None, compute over the\n        whole array `a`.\n    inclusive : {True, False}, optional\n        This flag determines whether values exactly equal to the upper limit\n        are included.  The default value is True.\n\n    Returns\n    -------\n    tmax : float, int or ndarray\n\n    Notes\n    -----\n    For more details on `tmax`, see `scipy.stats.tmax`.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats import mstats\n    >>> a = np.array([[6, 8, 3, 0],\n    ...               [3, 9, 1, 2],\n    ...               [8, 7, 8, 2],\n    ...               [5, 6, 0, 2],\n    ...               [4, 5, 5, 2]])\n    ...\n    ...\n    >>> mstats.tmax(a, 4)\n    masked_array(data=[4, --, 3, 2],\n                 mask=[False,  True, False, False],\n           fill_value=999999)\n\n    '
    (a, axis) = _chk_asarray(a, axis)
    am = trima(a, (None, upperlimit), (False, inclusive))
    return ma.maximum.reduce(am, axis)

def tsem(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the trimmed standard error of the mean.\n\n    This function finds the standard error of the mean for given\n    values, ignoring values outside the given `limits`.\n\n    Parameters\n    ----------\n    a : array_like\n        array of values\n    limits : None or (lower limit, upper limit), optional\n        Values in the input array less than the lower limit or greater than the\n        upper limit will be ignored. When limits is None, then all values are\n        used. Either of the limit values in the tuple can also be None\n        representing a half-open interval.  The default value is None.\n    inclusive : (bool, bool), optional\n        A tuple consisting of the (lower flag, upper flag).  These flags\n        determine whether values exactly equal to the lower or upper limits\n        are included.  The default value is (True, True).\n    axis : int or None, optional\n        Axis along which to operate. If None, compute over the\n        whole array. Default is zero.\n    ddof : int, optional\n        Delta degrees of freedom. Default is 1.\n\n    Returns\n    -------\n    tsem : float\n\n    Notes\n    -----\n    For more details on `tsem`, see `scipy.stats.tsem`.\n\n    '
    a = ma.asarray(a).ravel()
    if limits is None:
        n = float(a.count())
        return a.std(axis=axis, ddof=ddof) / ma.sqrt(n)
    am = trima(a.ravel(), limits, inclusive)
    sd = np.sqrt(am.var(axis=axis, ddof=ddof))
    return sd / np.sqrt(am.count())

def winsorize(a, limits=None, inclusive=(True, True), inplace=False, axis=None, nan_policy='propagate'):
    if False:
        while True:
            i = 10
    "Returns a Winsorized version of the input array.\n\n    The (limits[0])th lowest values are set to the (limits[0])th percentile,\n    and the (limits[1])th highest values are set to the (1 - limits[1])th\n    percentile.\n    Masked values are skipped.\n\n\n    Parameters\n    ----------\n    a : sequence\n        Input array.\n    limits : {None, tuple of float}, optional\n        Tuple of the percentages to cut on each side of the array, with respect\n        to the number of unmasked data, as floats between 0. and 1.\n        Noting n the number of unmasked data before trimming, the\n        (n*limits[0])th smallest data and the (n*limits[1])th largest data are\n        masked, and the total number of unmasked data after trimming\n        is n*(1.-sum(limits)) The value of one limit can be set to None to\n        indicate an open interval.\n    inclusive : {(True, True) tuple}, optional\n        Tuple indicating whether the number of data being masked on each side\n        should be truncated (True) or rounded (False).\n    inplace : {False, True}, optional\n        Whether to winsorize in place (True) or to use a copy (False)\n    axis : {None, int}, optional\n        Axis along which to trim. If None, the whole array is trimmed, but its\n        shape is maintained.\n    nan_policy : {'propagate', 'raise', 'omit'}, optional\n        Defines how to handle when input contains nan.\n        The following options are available (default is 'propagate'):\n\n          * 'propagate': allows nan values and may overwrite or propagate them\n          * 'raise': throws an error\n          * 'omit': performs the calculations ignoring nan values\n\n    Notes\n    -----\n    This function is applied to reduce the effect of possibly spurious outliers\n    by limiting the extreme values.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats.mstats import winsorize\n\n    A shuffled array contains integers from 1 to 10.\n\n    >>> a = np.array([10, 4, 9, 8, 5, 3, 7, 2, 1, 6])\n\n    The 10% of the lowest value (i.e., `1`) and the 20% of the highest\n    values (i.e., `9` and `10`) are replaced.\n\n    >>> winsorize(a, limits=[0.1, 0.2])\n    masked_array(data=[8, 4, 8, 8, 5, 3, 7, 2, 2, 6],\n                 mask=False,\n           fill_value=999999)\n\n    "

    def _winsorize1D(a, low_limit, up_limit, low_include, up_include, contains_nan, nan_policy):
        if False:
            for i in range(10):
                print('nop')
        n = a.count()
        idx = a.argsort()
        if contains_nan:
            nan_count = np.count_nonzero(np.isnan(a))
        if low_limit:
            if low_include:
                lowidx = int(low_limit * n)
            else:
                lowidx = np.round(low_limit * n).astype(int)
            if contains_nan and nan_policy == 'omit':
                lowidx = min(lowidx, n - nan_count - 1)
            a[idx[:lowidx]] = a[idx[lowidx]]
        if up_limit is not None:
            if up_include:
                upidx = n - int(n * up_limit)
            else:
                upidx = n - np.round(n * up_limit).astype(int)
            if contains_nan and nan_policy == 'omit':
                a[idx[upidx:-nan_count]] = a[idx[upidx - 1]]
            else:
                a[idx[upidx:]] = a[idx[upidx - 1]]
        return a
    (contains_nan, nan_policy) = _contains_nan(a, nan_policy)
    a = ma.array(a, copy=np.logical_not(inplace))
    if limits is None:
        return a
    if not isinstance(limits, tuple) and isinstance(limits, float):
        limits = (limits, limits)
    (lolim, uplim) = limits
    errmsg = 'The proportion to cut from the %s should be between 0. and 1.'
    if lolim is not None:
        if lolim > 1.0 or lolim < 0:
            raise ValueError(errmsg % 'beginning' + '(got %s)' % lolim)
    if uplim is not None:
        if uplim > 1.0 or uplim < 0:
            raise ValueError(errmsg % 'end' + '(got %s)' % uplim)
    (loinc, upinc) = inclusive
    if axis is None:
        shp = a.shape
        return _winsorize1D(a.ravel(), lolim, uplim, loinc, upinc, contains_nan, nan_policy).reshape(shp)
    else:
        return ma.apply_along_axis(_winsorize1D, axis, a, lolim, uplim, loinc, upinc, contains_nan, nan_policy)

def moment(a, moment=1, axis=0):
    if False:
        return 10
    '\n    Calculates the nth moment about the mean for a sample.\n\n    Parameters\n    ----------\n    a : array_like\n       data\n    moment : int, optional\n       order of central moment that is returned\n    axis : int or None, optional\n       Axis along which the central moment is computed. Default is 0.\n       If None, compute over the whole array `a`.\n\n    Returns\n    -------\n    n-th central moment : ndarray or float\n       The appropriate moment along the given axis or over all values if axis\n       is None. The denominator for the moment calculation is the number of\n       observations, no degrees of freedom correction is done.\n\n    Notes\n    -----\n    For more details about `moment`, see `scipy.stats.moment`.\n\n    '
    (a, axis) = _chk_asarray(a, axis)
    if a.size == 0:
        moment_shape = list(a.shape)
        del moment_shape[axis]
        dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64
        out_shape = moment_shape if np.isscalar(moment) else [len(moment)] + moment_shape
        if len(out_shape) == 0:
            return dtype(np.nan)
        else:
            return ma.array(np.full(out_shape, np.nan, dtype=dtype))
    if not np.isscalar(moment):
        mean = a.mean(axis, keepdims=True)
        mmnt = [_moment(a, i, axis, mean=mean) for i in moment]
        return ma.array(mmnt)
    else:
        return _moment(a, moment, axis)

def _moment(a, moment, axis, *, mean=None):
    if False:
        print('Hello World!')
    if np.abs(moment - np.round(moment)) > 0:
        raise ValueError('All moment parameters must be integers')
    if moment == 0 or moment == 1:
        shape = list(a.shape)
        del shape[axis]
        dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64
        if len(shape) == 0:
            return dtype(1.0 if moment == 0 else 0.0)
        else:
            return ma.ones(shape, dtype=dtype) if moment == 0 else ma.zeros(shape, dtype=dtype)
    else:
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)
        mean = a.mean(axis, keepdims=True) if mean is None else mean
        a_zero_mean = a - mean
        if n_list[-1] == 1:
            s = a_zero_mean.copy()
        else:
            s = a_zero_mean ** 2
        for n in n_list[-2::-1]:
            s = s ** 2
            if n % 2:
                s *= a_zero_mean
        return s.mean(axis)

def variation(a, axis=0, ddof=0):
    if False:
        return 10
    "\n    Compute the coefficient of variation.\n\n    The coefficient of variation is the standard deviation divided by the\n    mean.  This function is equivalent to::\n\n        np.std(x, axis=axis, ddof=ddof) / np.mean(x)\n\n    The default for ``ddof`` is 0, but many definitions of the coefficient\n    of variation use the square root of the unbiased sample variance\n    for the sample standard deviation, which corresponds to ``ddof=1``.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axis : int or None, optional\n        Axis along which to calculate the coefficient of variation. Default\n        is 0. If None, compute over the whole array `a`.\n    ddof : int, optional\n        Delta degrees of freedom.  Default is 0.\n\n    Returns\n    -------\n    variation : ndarray\n        The calculated variation along the requested axis.\n\n    Notes\n    -----\n    For more details about `variation`, see `scipy.stats.variation`.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats.mstats import variation\n    >>> a = np.array([2,8,4])\n    >>> variation(a)\n    0.5345224838248487\n    >>> b = np.array([2,8,3,4])\n    >>> c = np.ma.masked_array(b, mask=[0,0,1,0])\n    >>> variation(c)\n    0.5345224838248487\n\n    In the example above, it can be seen that this works the same as\n    `scipy.stats.variation` except 'stats.mstats.variation' ignores masked\n    array elements.\n\n    "
    (a, axis) = _chk_asarray(a, axis)
    return a.std(axis, ddof=ddof) / a.mean(axis)

def skew(a, axis=0, bias=True):
    if False:
        i = 10
        return i + 15
    '\n    Computes the skewness of a data set.\n\n    Parameters\n    ----------\n    a : ndarray\n        data\n    axis : int or None, optional\n        Axis along which skewness is calculated. Default is 0.\n        If None, compute over the whole array `a`.\n    bias : bool, optional\n        If False, then the calculations are corrected for statistical bias.\n\n    Returns\n    -------\n    skewness : ndarray\n        The skewness of values along an axis, returning 0 where all values are\n        equal.\n\n    Notes\n    -----\n    For more details about `skew`, see `scipy.stats.skew`.\n\n    '
    (a, axis) = _chk_asarray(a, axis)
    mean = a.mean(axis, keepdims=True)
    m2 = _moment(a, 2, axis, mean=mean)
    m3 = _moment(a, 3, axis, mean=mean)
    zero = m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(axis)) ** 2
    with np.errstate(all='ignore'):
        vals = ma.where(zero, 0, m3 / m2 ** 1.5)
    if not bias and zero is not ma.masked and (m2 is not ma.masked):
        n = a.count(axis)
        can_correct = ~zero & (n > 2)
        if can_correct.any():
            n = np.extract(can_correct, n)
            m2 = np.extract(can_correct, m2)
            m3 = np.extract(can_correct, m3)
            nval = ma.sqrt((n - 1.0) * n) / (n - 2.0) * m3 / m2 ** 1.5
            np.place(vals, can_correct, nval)
    return vals

def kurtosis(a, axis=0, fisher=True, bias=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Computes the kurtosis (Fisher or Pearson) of a dataset.\n\n    Kurtosis is the fourth central moment divided by the square of the\n    variance. If Fisher's definition is used, then 3.0 is subtracted from\n    the result to give 0.0 for a normal distribution.\n\n    If bias is False then the kurtosis is calculated using k statistics to\n    eliminate bias coming from biased moment estimators\n\n    Use `kurtosistest` to see if result is close enough to normal.\n\n    Parameters\n    ----------\n    a : array\n        data for which the kurtosis is calculated\n    axis : int or None, optional\n        Axis along which the kurtosis is calculated. Default is 0.\n        If None, compute over the whole array `a`.\n    fisher : bool, optional\n        If True, Fisher's definition is used (normal ==> 0.0). If False,\n        Pearson's definition is used (normal ==> 3.0).\n    bias : bool, optional\n        If False, then the calculations are corrected for statistical bias.\n\n    Returns\n    -------\n    kurtosis : array\n        The kurtosis of values along an axis. If all values are equal,\n        return -3 for Fisher's definition and 0 for Pearson's definition.\n\n    Notes\n    -----\n    For more details about `kurtosis`, see `scipy.stats.kurtosis`.\n\n    "
    (a, axis) = _chk_asarray(a, axis)
    mean = a.mean(axis, keepdims=True)
    m2 = _moment(a, 2, axis, mean=mean)
    m4 = _moment(a, 4, axis, mean=mean)
    zero = m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(axis)) ** 2
    with np.errstate(all='ignore'):
        vals = ma.where(zero, 0, m4 / m2 ** 2.0)
    if not bias and zero is not ma.masked and (m2 is not ma.masked):
        n = a.count(axis)
        can_correct = ~zero & (n > 3)
        if can_correct.any():
            n = np.extract(can_correct, n)
            m2 = np.extract(can_correct, m2)
            m4 = np.extract(can_correct, m4)
            nval = 1.0 / (n - 2) / (n - 3) * ((n * n - 1.0) * m4 / m2 ** 2.0 - 3 * (n - 1) ** 2.0)
            np.place(vals, can_correct, nval + 3.0)
    if fisher:
        return vals - 3
    else:
        return vals
DescribeResult = namedtuple('DescribeResult', ('nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis'))

def describe(a, axis=0, ddof=0, bias=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes several descriptive statistics of the passed array.\n\n    Parameters\n    ----------\n    a : array_like\n        Data array\n    axis : int or None, optional\n        Axis along which to calculate statistics. Default 0. If None,\n        compute over the whole array `a`.\n    ddof : int, optional\n        degree of freedom (default 0); note that default ddof is different\n        from the same routine in stats.describe\n    bias : bool, optional\n        If False, then the skewness and kurtosis calculations are corrected for\n        statistical bias.\n\n    Returns\n    -------\n    nobs : int\n        (size of the data (discarding missing values)\n\n    minmax : (int, int)\n        min, max\n\n    mean : float\n        arithmetic mean\n\n    variance : float\n        unbiased variance\n\n    skewness : float\n        biased skewness\n\n    kurtosis : float\n        biased kurtosis\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats.mstats import describe\n    >>> ma = np.ma.array(range(6), mask=[0, 0, 0, 1, 1, 1])\n    >>> describe(ma)\n    DescribeResult(nobs=3, minmax=(masked_array(data=0,\n                 mask=False,\n           fill_value=999999), masked_array(data=2,\n                 mask=False,\n           fill_value=999999)), mean=1.0, variance=0.6666666666666666,\n           skewness=masked_array(data=0., mask=False, fill_value=1e+20),\n            kurtosis=-1.5)\n\n    '
    (a, axis) = _chk_asarray(a, axis)
    n = a.count(axis)
    mm = (ma.minimum.reduce(a, axis=axis), ma.maximum.reduce(a, axis=axis))
    m = a.mean(axis)
    v = a.var(axis, ddof=ddof)
    sk = skew(a, axis, bias=bias)
    kurt = kurtosis(a, axis, bias=bias)
    return DescribeResult(n, mm, m, v, sk, kurt)

def stde_median(data, axis=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns the McKean-Schrader estimate of the standard error of the sample\n    median along the given axis. masked values are discarded.\n\n    Parameters\n    ----------\n    data : ndarray\n        Data to trim.\n    axis : {None,int}, optional\n        Axis along which to perform the trimming.\n        If None, the input array is first flattened.\n\n    '

    def _stdemed_1D(data):
        if False:
            while True:
                i = 10
        data = np.sort(data.compressed())
        n = len(data)
        z = 2.5758293035489004
        k = int(np.round((n + 1) / 2.0 - z * np.sqrt(n / 4.0), 0))
        return (data[n - k] - data[k - 1]) / (2.0 * z)
    data = ma.array(data, copy=False, subok=True)
    if axis is None:
        return _stdemed_1D(data)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, but got data.ndim = %d" % data.ndim)
        return ma.apply_along_axis(_stdemed_1D, axis, data)
SkewtestResult = namedtuple('SkewtestResult', ('statistic', 'pvalue'))

def skewtest(a, axis=0, alternative='two-sided'):
    if False:
        return 10
    "\n    Tests whether the skew is different from the normal distribution.\n\n    Parameters\n    ----------\n    a : array_like\n        The data to be tested\n    axis : int or None, optional\n       Axis along which statistics are calculated. Default is 0.\n       If None, compute over the whole array `a`.\n    alternative : {'two-sided', 'less', 'greater'}, optional\n        Defines the alternative hypothesis. Default is 'two-sided'.\n        The following options are available:\n\n        * 'two-sided': the skewness of the distribution underlying the sample\n          is different from that of the normal distribution (i.e. 0)\n        * 'less': the skewness of the distribution underlying the sample\n          is less than that of the normal distribution\n        * 'greater': the skewness of the distribution underlying the sample\n          is greater than that of the normal distribution\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    statistic : array_like\n        The computed z-score for this test.\n    pvalue : array_like\n        A p-value for the hypothesis test\n\n    Notes\n    -----\n    For more details about `skewtest`, see `scipy.stats.skewtest`.\n\n    "
    (a, axis) = _chk_asarray(a, axis)
    if axis is None:
        a = a.ravel()
        axis = 0
    b2 = skew(a, axis)
    n = a.count(axis)
    if np.min(n) < 8:
        raise ValueError('skewtest is not valid with less than 8 samples; %i samples were given.' % np.min(n))
    y = b2 * ma.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
    W2 = -1 + ma.sqrt(2 * (beta2 - 1))
    delta = 1 / ma.sqrt(0.5 * ma.log(W2))
    alpha = ma.sqrt(2.0 / (W2 - 1))
    y = ma.where(y == 0, 1, y)
    Z = delta * ma.log(y / alpha + ma.sqrt((y / alpha) ** 2 + 1))
    return SkewtestResult(*scipy.stats._stats_py._normtest_finish(Z, alternative))
KurtosistestResult = namedtuple('KurtosistestResult', ('statistic', 'pvalue'))

def kurtosistest(a, axis=0, alternative='two-sided'):
    if False:
        while True:
            i = 10
    "\n    Tests whether a dataset has normal kurtosis\n\n    Parameters\n    ----------\n    a : array_like\n        array of the sample data\n    axis : int or None, optional\n       Axis along which to compute test. Default is 0. If None,\n       compute over the whole array `a`.\n    alternative : {'two-sided', 'less', 'greater'}, optional\n        Defines the alternative hypothesis.\n        The following options are available (default is 'two-sided'):\n\n        * 'two-sided': the kurtosis of the distribution underlying the sample\n          is different from that of the normal distribution\n        * 'less': the kurtosis of the distribution underlying the sample\n          is less than that of the normal distribution\n        * 'greater': the kurtosis of the distribution underlying the sample\n          is greater than that of the normal distribution\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    statistic : array_like\n        The computed z-score for this test.\n    pvalue : array_like\n        The p-value for the hypothesis test\n\n    Notes\n    -----\n    For more details about `kurtosistest`, see `scipy.stats.kurtosistest`.\n\n    "
    (a, axis) = _chk_asarray(a, axis)
    n = a.count(axis=axis)
    if np.min(n) < 5:
        raise ValueError('kurtosistest requires at least 5 observations; %i observations were given.' % np.min(n))
    if np.min(n) < 20:
        warnings.warn('kurtosistest only valid for n>=20 ... continuing anyway, n=%i' % np.min(n))
    b2 = kurtosis(a, axis, fisher=False)
    E = 3.0 * (n - 1) / (n + 1)
    varb2 = 24.0 * n * (n - 2.0) * (n - 3) / ((n + 1) * (n + 1.0) * (n + 3) * (n + 5))
    x = (b2 - E) / ma.sqrt(varb2)
    sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * np.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    A = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + np.sqrt(1 + 4.0 / sqrtbeta1 ** 2))
    term1 = 1 - 2.0 / (9.0 * A)
    denom = 1 + x * ma.sqrt(2 / (A - 4.0))
    if np.ma.isMaskedArray(denom):
        denom[denom == 0.0] = masked
    elif denom == 0.0:
        denom = masked
    term2 = np.ma.where(denom > 0, ma.power((1 - 2.0 / A) / denom, 1 / 3.0), -ma.power(-(1 - 2.0 / A) / denom, 1 / 3.0))
    Z = (term1 - term2) / np.sqrt(2 / (9.0 * A))
    return KurtosistestResult(*scipy.stats._stats_py._normtest_finish(Z, alternative))
NormaltestResult = namedtuple('NormaltestResult', ('statistic', 'pvalue'))

def normaltest(a, axis=0):
    if False:
        print('Hello World!')
    '\n    Tests whether a sample differs from a normal distribution.\n\n    Parameters\n    ----------\n    a : array_like\n        The array containing the data to be tested.\n    axis : int or None, optional\n        Axis along which to compute test. Default is 0. If None,\n        compute over the whole array `a`.\n\n    Returns\n    -------\n    statistic : float or array\n        ``s^2 + k^2``, where ``s`` is the z-score returned by `skewtest` and\n        ``k`` is the z-score returned by `kurtosistest`.\n    pvalue : float or array\n       A 2-sided chi squared probability for the hypothesis test.\n\n    Notes\n    -----\n    For more details about `normaltest`, see `scipy.stats.normaltest`.\n\n    '
    (a, axis) = _chk_asarray(a, axis)
    (s, _) = skewtest(a, axis)
    (k, _) = kurtosistest(a, axis)
    k2 = s * s + k * k
    return NormaltestResult(k2, distributions.chi2.sf(k2, 2))

def mquantiles(a, prob=list([0.25, 0.5, 0.75]), alphap=0.4, betap=0.4, axis=None, limit=()):
    if False:
        i = 10
        return i + 15
    '\n    Computes empirical quantiles for a data array.\n\n    Samples quantile are defined by ``Q(p) = (1-gamma)*x[j] + gamma*x[j+1]``,\n    where ``x[j]`` is the j-th order statistic, and gamma is a function of\n    ``j = floor(n*p + m)``, ``m = alphap + p*(1 - alphap - betap)`` and\n    ``g = n*p + m - j``.\n\n    Reinterpreting the above equations to compare to **R** lead to the\n    equation: ``p(k) = (k - alphap)/(n + 1 - alphap - betap)``\n\n    Typical values of (alphap,betap) are:\n        - (0,1)    : ``p(k) = k/n`` : linear interpolation of cdf\n          (**R** type 4)\n        - (.5,.5)  : ``p(k) = (k - 1/2.)/n`` : piecewise linear function\n          (**R** type 5)\n        - (0,0)    : ``p(k) = k/(n+1)`` :\n          (**R** type 6)\n        - (1,1)    : ``p(k) = (k-1)/(n-1)``: p(k) = mode[F(x[k])].\n          (**R** type 7, **R** default)\n        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``: Then p(k) ~ median[F(x[k])].\n          The resulting quantile estimates are approximately median-unbiased\n          regardless of the distribution of x.\n          (**R** type 8)\n        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``: Blom.\n          The resulting quantile estimates are approximately unbiased\n          if x is normally distributed\n          (**R** type 9)\n        - (.4,.4)  : approximately quantile unbiased (Cunnane)\n        - (.35,.35): APL, used with PWM\n\n    Parameters\n    ----------\n    a : array_like\n        Input data, as a sequence or array of dimension at most 2.\n    prob : array_like, optional\n        List of quantiles to compute.\n    alphap : float, optional\n        Plotting positions parameter, default is 0.4.\n    betap : float, optional\n        Plotting positions parameter, default is 0.4.\n    axis : int, optional\n        Axis along which to perform the trimming.\n        If None (default), the input array is first flattened.\n    limit : tuple, optional\n        Tuple of (lower, upper) values.\n        Values of `a` outside this open interval are ignored.\n\n    Returns\n    -------\n    mquantiles : MaskedArray\n        An array containing the calculated quantiles.\n\n    Notes\n    -----\n    This formulation is very similar to **R** except the calculation of\n    ``m`` from ``alphap`` and ``betap``, where in **R** ``m`` is defined\n    with each type.\n\n    References\n    ----------\n    .. [1] *R* statistical software: https://www.r-project.org/\n    .. [2] *R* ``quantile`` function:\n            http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats.mstats import mquantiles\n    >>> a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])\n    >>> mquantiles(a)\n    array([ 19.2,  40. ,  42.8])\n\n    Using a 2D array, specifying axis and limit.\n\n    >>> data = np.array([[   6.,    7.,    1.],\n    ...                  [  47.,   15.,    2.],\n    ...                  [  49.,   36.,    3.],\n    ...                  [  15.,   39.,    4.],\n    ...                  [  42.,   40., -999.],\n    ...                  [  41.,   41., -999.],\n    ...                  [   7., -999., -999.],\n    ...                  [  39., -999., -999.],\n    ...                  [  43., -999., -999.],\n    ...                  [  40., -999., -999.],\n    ...                  [  36., -999., -999.]])\n    >>> print(mquantiles(data, axis=0, limit=(0, 50)))\n    [[19.2  14.6   1.45]\n     [40.   37.5   2.5 ]\n     [42.8  40.05  3.55]]\n\n    >>> data[:, 2] = -999.\n    >>> print(mquantiles(data, axis=0, limit=(0, 50)))\n    [[19.200000000000003 14.6 --]\n     [40.0 37.5 --]\n     [42.800000000000004 40.05 --]]\n\n    '

    def _quantiles1D(data, m, p):
        if False:
            while True:
                i = 10
        x = np.sort(data.compressed())
        n = len(x)
        if n == 0:
            return ma.array(np.empty(len(p), dtype=float), mask=True)
        elif n == 1:
            return ma.array(np.resize(x, p.shape), mask=nomask)
        aleph = n * p + m
        k = np.floor(aleph.clip(1, n - 1)).astype(int)
        gamma = (aleph - k).clip(0, 1)
        return (1.0 - gamma) * x[(k - 1).tolist()] + gamma * x[k.tolist()]
    data = ma.array(a, copy=False)
    if data.ndim > 2:
        raise TypeError('Array should be 2D at most !')
    if limit:
        condition = (limit[0] < data) & (data < limit[1])
        data[~condition.filled(True)] = masked
    p = np.array(prob, copy=False, ndmin=1)
    m = alphap + p * (1.0 - alphap - betap)
    if axis is None:
        return _quantiles1D(data, m, p)
    return ma.apply_along_axis(_quantiles1D, axis, data, m, p)

def scoreatpercentile(data, per, limit=(), alphap=0.4, betap=0.4):
    if False:
        return 10
    "Calculate the score at the given 'per' percentile of the\n    sequence a.  For example, the score at per=50 is the median.\n\n    This function is a shortcut to mquantile\n\n    "
    if per < 0 or per > 100.0:
        raise ValueError('The percentile should be between 0. and 100. ! (got %s)' % per)
    return mquantiles(data, prob=[per / 100.0], alphap=alphap, betap=betap, limit=limit, axis=0).squeeze()

def plotting_positions(data, alpha=0.4, beta=0.4):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns plotting positions (or empirical percentile points) for the data.\n\n    Plotting positions are defined as ``(i-alpha)/(n+1-alpha-beta)``, where:\n        - i is the rank order statistics\n        - n is the number of unmasked values along the given axis\n        - `alpha` and `beta` are two parameters.\n\n    Typical values for `alpha` and `beta` are:\n        - (0,1)    : ``p(k) = k/n``, linear interpolation of cdf (R, type 4)\n        - (.5,.5)  : ``p(k) = (k-1/2.)/n``, piecewise linear function\n          (R, type 5)\n        - (0,0)    : ``p(k) = k/(n+1)``, Weibull (R type 6)\n        - (1,1)    : ``p(k) = (k-1)/(n-1)``, in this case,\n          ``p(k) = mode[F(x[k])]``. That's R default (R type 7)\n        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``, then\n          ``p(k) ~ median[F(x[k])]``.\n          The resulting quantile estimates are approximately median-unbiased\n          regardless of the distribution of x. (R type 8)\n        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``, Blom.\n          The resulting quantile estimates are approximately unbiased\n          if x is normally distributed (R type 9)\n        - (.4,.4)  : approximately quantile unbiased (Cunnane)\n        - (.35,.35): APL, used with PWM\n        - (.3175, .3175): used in scipy.stats.probplot\n\n    Parameters\n    ----------\n    data : array_like\n        Input data, as a sequence or array of dimension at most 2.\n    alpha : float, optional\n        Plotting positions parameter. Default is 0.4.\n    beta : float, optional\n        Plotting positions parameter. Default is 0.4.\n\n    Returns\n    -------\n    positions : MaskedArray\n        The calculated plotting positions.\n\n    "
    data = ma.array(data, copy=False).reshape(1, -1)
    n = data.count()
    plpos = np.empty(data.size, dtype=float)
    plpos[n:] = 0
    plpos[data.argsort(axis=None)[:n]] = (np.arange(1, n + 1) - alpha) / (n + 1.0 - alpha - beta)
    return ma.array(plpos, mask=data._mask)
meppf = plotting_positions

def obrientransform(*args):
    if False:
        while True:
            i = 10
    '\n    Computes a transform on input data (any number of columns).  Used to\n    test for homogeneity of variance prior to running one-way stats.  Each\n    array in ``*args`` is one level of a factor.  If an `f_oneway()` run on\n    the transformed data and found significant, variances are unequal.   From\n    Maxwell and Delaney, p.112.\n\n    Returns: transformed data for use in an ANOVA\n    '
    data = argstoarray(*args).T
    v = data.var(axis=0, ddof=1)
    m = data.mean(0)
    n = data.count(0).astype(float)
    data -= m
    data **= 2
    data *= (n - 1.5) * n
    data -= 0.5 * v * (n - 1)
    data /= (n - 1.0) * (n - 2.0)
    if not ma.allclose(v, data.mean(0)):
        raise ValueError('Lack of convergence in obrientransform.')
    return data

def sem(a, axis=0, ddof=1):
    if False:
        i = 10
        return i + 15
    '\n    Calculates the standard error of the mean of the input array.\n\n    Also sometimes called standard error of measurement.\n\n    Parameters\n    ----------\n    a : array_like\n        An array containing the values for which the standard error is\n        returned.\n    axis : int or None, optional\n        If axis is None, ravel `a` first. If axis is an integer, this will be\n        the axis over which to operate. Defaults to 0.\n    ddof : int, optional\n        Delta degrees-of-freedom. How many degrees of freedom to adjust\n        for bias in limited samples relative to the population estimate\n        of variance. Defaults to 1.\n\n    Returns\n    -------\n    s : ndarray or float\n        The standard error of the mean in the sample(s), along the input axis.\n\n    Notes\n    -----\n    The default value for `ddof` changed in scipy 0.15.0 to be consistent with\n    `scipy.stats.sem` as well as with the most common definition used (like in\n    the R documentation).\n\n    Examples\n    --------\n    Find standard error along the first axis:\n\n    >>> import numpy as np\n    >>> from scipy import stats\n    >>> a = np.arange(20).reshape(5,4)\n    >>> print(stats.mstats.sem(a))\n    [2.8284271247461903 2.8284271247461903 2.8284271247461903\n     2.8284271247461903]\n\n    Find standard error across the whole array, using n degrees of freedom:\n\n    >>> print(stats.mstats.sem(a, axis=None, ddof=0))\n    1.2893796958227628\n\n    '
    (a, axis) = _chk_asarray(a, axis)
    n = a.count(axis=axis)
    s = a.std(axis=axis, ddof=ddof) / ma.sqrt(n)
    return s
F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))

def f_oneway(*args):
    if False:
        return 10
    '\n    Performs a 1-way ANOVA, returning an F-value and probability given\n    any number of groups.  From Heiman, pp.394-7.\n\n    Usage: ``f_oneway(*args)``, where ``*args`` is 2 or more arrays,\n    one per treatment group.\n\n    Returns\n    -------\n    statistic : float\n        The computed F-value of the test.\n    pvalue : float\n        The associated p-value from the F-distribution.\n\n    '
    data = argstoarray(*args)
    ngroups = len(data)
    ntot = data.count()
    sstot = (data ** 2).sum() - data.sum() ** 2 / float(ntot)
    ssbg = (data.count(-1) * (data.mean(-1) - data.mean()) ** 2).sum()
    sswg = sstot - ssbg
    dfbg = ngroups - 1
    dfwg = ntot - ngroups
    msb = ssbg / float(dfbg)
    msw = sswg / float(dfwg)
    f = msb / msw
    prob = special.fdtrc(dfbg, dfwg, f)
    return F_onewayResult(f, prob)
FriedmanchisquareResult = namedtuple('FriedmanchisquareResult', ('statistic', 'pvalue'))

def friedmanchisquare(*args):
    if False:
        for i in range(10):
            print('nop')
    'Friedman Chi-Square is a non-parametric, one-way within-subjects ANOVA.\n    This function calculates the Friedman Chi-square test for repeated measures\n    and returns the result, along with the associated probability value.\n\n    Each input is considered a given group. Ideally, the number of treatments\n    among each group should be equal. If this is not the case, only the first\n    n treatments are taken into account, where n is the number of treatments\n    of the smallest group.\n    If a group has some missing values, the corresponding treatments are masked\n    in the other groups.\n    The test statistic is corrected for ties.\n\n    Masked values in one group are propagated to the other groups.\n\n    Returns\n    -------\n    statistic : float\n        the test statistic.\n    pvalue : float\n        the associated p-value.\n\n    '
    data = argstoarray(*args).astype(float)
    k = len(data)
    if k < 3:
        raise ValueError('Less than 3 groups (%i): ' % k + 'the Friedman test is NOT appropriate.')
    ranked = ma.masked_values(rankdata(data, axis=0), 0)
    if ranked._mask is not nomask:
        ranked = ma.mask_cols(ranked)
        ranked = ranked.compressed().reshape(k, -1).view(ndarray)
    else:
        ranked = ranked._data
    (k, n) = ranked.shape
    repeats = [find_repeats(row) for row in ranked.T]
    ties = np.array([y for (x, y) in repeats if x.size > 0])
    tie_correction = 1 - (ties ** 3 - ties).sum() / float(n * (k ** 3 - k))
    ssbg = np.sum((ranked.sum(-1) - n * (k + 1) / 2.0) ** 2)
    chisq = ssbg * 12.0 / (n * k * (k + 1)) * 1.0 / tie_correction
    return FriedmanchisquareResult(chisq, distributions.chi2.sf(chisq, k - 1))
BrunnerMunzelResult = namedtuple('BrunnerMunzelResult', ('statistic', 'pvalue'))

def brunnermunzel(x, y, alternative='two-sided', distribution='t'):
    if False:
        return 10
    "\n    Computes the Brunner-Munzel test on samples x and y\n\n    Missing values in `x` and/or `y` are discarded.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Array of samples, should be one-dimensional.\n    alternative : 'less', 'two-sided', or 'greater', optional\n        Whether to get the p-value for the one-sided hypothesis ('less'\n        or 'greater') or for the two-sided hypothesis ('two-sided').\n        Defaults value is 'two-sided' .\n    distribution : 't' or 'normal', optional\n        Whether to get the p-value by t-distribution or by standard normal\n        distribution.\n        Defaults value is 't' .\n\n    Returns\n    -------\n    statistic : float\n        The Brunner-Munzer W statistic.\n    pvalue : float\n        p-value assuming an t distribution. One-sided or\n        two-sided, depending on the choice of `alternative` and `distribution`.\n\n    See Also\n    --------\n    mannwhitneyu : Mann-Whitney rank test on two samples.\n\n    Notes\n    -----\n    For more details on `brunnermunzel`, see `scipy.stats.brunnermunzel`.\n\n    "
    x = ma.asarray(x).compressed().view(ndarray)
    y = ma.asarray(y).compressed().view(ndarray)
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return BrunnerMunzelResult(np.nan, np.nan)
    rankc = rankdata(np.concatenate((x, y)))
    rankcx = rankc[0:nx]
    rankcy = rankc[nx:nx + ny]
    rankcx_mean = np.mean(rankcx)
    rankcy_mean = np.mean(rankcy)
    rankx = rankdata(x)
    ranky = rankdata(y)
    rankx_mean = np.mean(rankx)
    ranky_mean = np.mean(ranky)
    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0))
    Sx /= nx - 1
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0))
    Sy /= ny - 1
    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)
    if distribution == 't':
        df_numer = np.power(nx * Sx + ny * Sy, 2.0)
        df_denom = np.power(nx * Sx, 2.0) / (nx - 1)
        df_denom += np.power(ny * Sy, 2.0) / (ny - 1)
        df = df_numer / df_denom
        p = distributions.t.cdf(wbfn, df)
    elif distribution == 'normal':
        p = distributions.norm.cdf(wbfn)
    else:
        raise ValueError("distribution should be 't' or 'normal'")
    if alternative == 'greater':
        pass
    elif alternative == 'less':
        p = 1 - p
    elif alternative == 'two-sided':
        p = 2 * np.min([p, 1 - p])
    else:
        raise ValueError("alternative should be 'less', 'greater' or 'two-sided'")
    return BrunnerMunzelResult(wbfn, p)