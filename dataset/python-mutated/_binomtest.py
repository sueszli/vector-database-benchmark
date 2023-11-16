from math import sqrt
import numpy as np
from scipy._lib._util import _validate_int
from scipy.optimize import brentq
from scipy.special import ndtri
from ._discrete_distns import binom
from ._common import ConfidenceInterval

class BinomTestResult:
    """
    Result of `scipy.stats.binomtest`.

    Attributes
    ----------
    k : int
        The number of successes (copied from `binomtest` input).
    n : int
        The number of trials (copied from `binomtest` input).
    alternative : str
        Indicates the alternative hypothesis specified in the input
        to `binomtest`.  It will be one of ``'two-sided'``, ``'greater'``,
        or ``'less'``.
    statistic: float
        The estimate of the proportion of successes.
    pvalue : float
        The p-value of the hypothesis test.

    """

    def __init__(self, k, n, alternative, statistic, pvalue):
        if False:
            while True:
                i = 10
        self.k = k
        self.n = n
        self.alternative = alternative
        self.statistic = statistic
        self.pvalue = pvalue
        self.proportion_estimate = statistic

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        s = f'BinomTestResult(k={self.k}, n={self.n}, alternative={self.alternative!r}, statistic={self.statistic}, pvalue={self.pvalue})'
        return s

    def proportion_ci(self, confidence_level=0.95, method='exact'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Compute the confidence interval for ``statistic``.\n\n        Parameters\n        ----------\n        confidence_level : float, optional\n            Confidence level for the computed confidence interval\n            of the estimated proportion. Default is 0.95.\n        method : {'exact', 'wilson', 'wilsoncc'}, optional\n            Selects the method used to compute the confidence interval\n            for the estimate of the proportion:\n\n            'exact' :\n                Use the Clopper-Pearson exact method [1]_.\n            'wilson' :\n                Wilson's method, without continuity correction ([2]_, [3]_).\n            'wilsoncc' :\n                Wilson's method, with continuity correction ([2]_, [3]_).\n\n            Default is ``'exact'``.\n\n        Returns\n        -------\n        ci : ``ConfidenceInterval`` object\n            The object has attributes ``low`` and ``high`` that hold the\n            lower and upper bounds of the confidence interval.\n\n        References\n        ----------\n        .. [1] C. J. Clopper and E. S. Pearson, The use of confidence or\n               fiducial limits illustrated in the case of the binomial,\n               Biometrika, Vol. 26, No. 4, pp 404-413 (Dec. 1934).\n        .. [2] E. B. Wilson, Probable inference, the law of succession, and\n               statistical inference, J. Amer. Stat. Assoc., 22, pp 209-212\n               (1927).\n        .. [3] Robert G. Newcombe, Two-sided confidence intervals for the\n               single proportion: comparison of seven methods, Statistics\n               in Medicine, 17, pp 857-872 (1998).\n\n        Examples\n        --------\n        >>> from scipy.stats import binomtest\n        >>> result = binomtest(k=7, n=50, p=0.1)\n        >>> result.statistic\n        0.14\n        >>> result.proportion_ci()\n        ConfidenceInterval(low=0.05819170033997342, high=0.26739600249700846)\n        "
        if method not in ('exact', 'wilson', 'wilsoncc'):
            raise ValueError("method must be one of 'exact', 'wilson' or 'wilsoncc'.")
        if not 0 <= confidence_level <= 1:
            raise ValueError('confidence_level must be in the interval [0, 1].')
        if method == 'exact':
            (low, high) = _binom_exact_conf_int(self.k, self.n, confidence_level, self.alternative)
        else:
            (low, high) = _binom_wilson_conf_int(self.k, self.n, confidence_level, self.alternative, correction=method == 'wilsoncc')
        return ConfidenceInterval(low=low, high=high)

def _findp(func):
    if False:
        return 10
    try:
        p = brentq(func, 0, 1)
    except RuntimeError:
        raise RuntimeError('numerical solver failed to converge when computing the confidence limits') from None
    except ValueError as exc:
        raise ValueError('brentq raised a ValueError; report this to the SciPy developers') from exc
    return p

def _binom_exact_conf_int(k, n, confidence_level, alternative):
    if False:
        i = 10
        return i + 15
    '\n    Compute the estimate and confidence interval for the binomial test.\n\n    Returns proportion, prop_low, prop_high\n    '
    if alternative == 'two-sided':
        alpha = (1 - confidence_level) / 2
        if k == 0:
            plow = 0.0
        else:
            plow = _findp(lambda p: binom.sf(k - 1, n, p) - alpha)
        if k == n:
            phigh = 1.0
        else:
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'less':
        alpha = 1 - confidence_level
        plow = 0.0
        if k == n:
            phigh = 1.0
        else:
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'greater':
        alpha = 1 - confidence_level
        if k == 0:
            plow = 0.0
        else:
            plow = _findp(lambda p: binom.sf(k - 1, n, p) - alpha)
        phigh = 1.0
    return (plow, phigh)

def _binom_wilson_conf_int(k, n, confidence_level, alternative, correction):
    if False:
        for i in range(10):
            print('nop')
    p = k / n
    if alternative == 'two-sided':
        z = ndtri(0.5 + 0.5 * confidence_level)
    else:
        z = ndtri(confidence_level)
    denom = 2 * (n + z ** 2)
    center = (2 * n * p + z ** 2) / denom
    q = 1 - p
    if correction:
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            dlo = (1 + z * sqrt(z ** 2 - 2 - 1 / n + 4 * p * (n * q + 1))) / denom
            lo = center - dlo
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            dhi = (1 + z * sqrt(z ** 2 + 2 - 1 / n + 4 * p * (n * q - 1))) / denom
            hi = center + dhi
    else:
        delta = z / denom * sqrt(4 * n * p * q + z ** 2)
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            lo = center - delta
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            hi = center + delta
    return (lo, hi)

def binomtest(k, n, p=0.5, alternative='two-sided'):
    if False:
        print('Hello World!')
    "\n    Perform a test that the probability of success is p.\n\n    The binomial test [1]_ is a test of the null hypothesis that the\n    probability of success in a Bernoulli experiment is `p`.\n\n    Details of the test can be found in many texts on statistics, such\n    as section 24.5 of [2]_.\n\n    Parameters\n    ----------\n    k : int\n        The number of successes.\n    n : int\n        The number of trials.\n    p : float, optional\n        The hypothesized probability of success, i.e. the expected\n        proportion of successes.  The value must be in the interval\n        ``0 <= p <= 1``. The default value is ``p = 0.5``.\n    alternative : {'two-sided', 'greater', 'less'}, optional\n        Indicates the alternative hypothesis. The default value is\n        'two-sided'.\n\n    Returns\n    -------\n    result : `~scipy.stats._result_classes.BinomTestResult` instance\n        The return value is an object with the following attributes:\n\n        k : int\n            The number of successes (copied from `binomtest` input).\n        n : int\n            The number of trials (copied from `binomtest` input).\n        alternative : str\n            Indicates the alternative hypothesis specified in the input\n            to `binomtest`.  It will be one of ``'two-sided'``, ``'greater'``,\n            or ``'less'``.\n        statistic : float\n            The estimate of the proportion of successes.\n        pvalue : float\n            The p-value of the hypothesis test.\n\n        The object has the following methods:\n\n        proportion_ci(confidence_level=0.95, method='exact') :\n            Compute the confidence interval for ``statistic``.\n\n    Notes\n    -----\n    .. versionadded:: 1.7.0\n\n    References\n    ----------\n    .. [1] Binomial test, https://en.wikipedia.org/wiki/Binomial_test\n    .. [2] Jerrold H. Zar, Biostatistical Analysis (fifth edition),\n           Prentice Hall, Upper Saddle River, New Jersey USA (2010)\n\n    Examples\n    --------\n    >>> from scipy.stats import binomtest\n\n    A car manufacturer claims that no more than 10% of their cars are unsafe.\n    15 cars are inspected for safety, 3 were found to be unsafe. Test the\n    manufacturer's claim:\n\n    >>> result = binomtest(3, n=15, p=0.1, alternative='greater')\n    >>> result.pvalue\n    0.18406106910639114\n\n    The null hypothesis cannot be rejected at the 5% level of significance\n    because the returned p-value is greater than the critical value of 5%.\n\n    The test statistic is equal to the estimated proportion, which is simply\n    ``3/15``:\n\n    >>> result.statistic\n    0.2\n\n    We can use the `proportion_ci()` method of the result to compute the\n    confidence interval of the estimate:\n\n    >>> result.proportion_ci(confidence_level=0.95)\n    ConfidenceInterval(low=0.05684686759024681, high=1.0)\n\n    "
    k = _validate_int(k, 'k', minimum=0)
    n = _validate_int(n, 'n', minimum=1)
    if k > n:
        raise ValueError('k must not be greater than n.')
    if not 0 <= p <= 1:
        raise ValueError('p must be in range [0,1]')
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized; \nmust be 'two-sided', 'less' or 'greater'")
    if alternative == 'less':
        pval = binom.cdf(k, n, p)
    elif alternative == 'greater':
        pval = binom.sf(k - 1, n, p)
    else:
        d = binom.pmf(k, n, p)
        rerr = 1 + 1e-07
        if k == p * n:
            pval = 1.0
        elif k < p * n:
            ix = _binary_search_for_binom_tst(lambda x1: -binom.pmf(x1, n, p), -d * rerr, np.ceil(p * n), n)
            y = n - ix + int(d * rerr == binom.pmf(ix, n, p))
            pval = binom.cdf(k, n, p) + binom.sf(n - y, n, p)
        else:
            ix = _binary_search_for_binom_tst(lambda x1: binom.pmf(x1, n, p), d * rerr, 0, np.floor(p * n))
            y = ix + 1
            pval = binom.cdf(y - 1, n, p) + binom.sf(k - 1, n, p)
        pval = min(1.0, pval)
    result = BinomTestResult(k=k, n=n, alternative=alternative, statistic=k / n, pvalue=pval)
    return result

def _binary_search_for_binom_tst(a, d, lo, hi):
    if False:
        return 10
    '\n    Conducts an implicit binary search on a function specified by `a`.\n\n    Meant to be used on the binomial PMF for the case of two-sided tests\n    to obtain the value on the other side of the mode where the tail\n    probability should be computed. The values on either side of\n    the mode are always in order, meaning binary search is applicable.\n\n    Parameters\n    ----------\n    a : callable\n      The function over which to perform binary search. Its values\n      for inputs lo and hi should be in ascending order.\n    d : float\n      The value to search.\n    lo : int\n      The lower end of range to search.\n    hi : int\n      The higher end of the range to search.\n\n    Returns\n    -------\n    int\n      The index, i between lo and hi\n      such that a(i)<=d<a(i+1)\n    '
    while lo < hi:
        mid = lo + (hi - lo) // 2
        midval = a(mid)
        if midval < d:
            lo = mid + 1
        elif midval > d:
            hi = mid - 1
        else:
            return mid
    if a(lo) <= d:
        return lo
    else:
        return lo - 1