"""
Tests and Confidence Intervals for Binomial Proportions

Created on Fri Mar 01 00:23:07 2013

Author: Josef Perktold
License: BSD-3
"""
from statsmodels.compat.python import lzip
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
FLOAT_INFO = np.finfo(float)

def _bound_proportion_confint(func: Callable[[float], float], qi: float, lower: bool=True) -> float:
    if False:
        while True:
            i = 10
    '\n    Try hard to find a bound different from eps/1 - eps in proportion_confint\n\n    Parameters\n    ----------\n    func : callable\n        Callable function to use as the objective of the search\n    qi : float\n        The empirical success rate\n    lower : bool\n        Whether to fund a lower bound for the left side of the CI\n\n    Returns\n    -------\n    float\n        The coarse bound\n    '
    default = FLOAT_INFO.eps if lower else 1.0 - FLOAT_INFO.eps

    def step(v):
        if False:
            return 10
        return v / 8 if lower else v + (1.0 - v) / 8
    x = step(qi)
    w = func(x)
    cnt = 1
    while w > 0 and cnt < 10:
        x = step(x)
        w = func(x)
        cnt += 1
    return x if cnt < 10 else default

def _bisection_search_conservative(func: Callable[[float], float], lb: float, ub: float, steps: int=27) -> Tuple[float, float]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Private function used as a fallback by proportion_confint\n\n    Used when brentq returns a non-conservative bound for the CI\n\n    Parameters\n    ----------\n    func : callable\n        Callable function to use as the objective of the search\n    lb : float\n        Lower bound\n    ub : float\n        Upper bound\n    steps : int\n        Number of steps to use in the bisection\n\n    Returns\n    -------\n    est : float\n        The estimated value.  Will always produce a negative value of func\n    func_val : float\n        The value of the function at the estimate\n    '
    upper = func(ub)
    lower = func(lb)
    best = upper if upper < 0 else lower
    best_pt = ub if upper < 0 else lb
    if np.sign(lower) == np.sign(upper):
        raise ValueError('problem with signs')
    mp = (ub + lb) / 2
    mid = func(mp)
    if mid < 0 and mid > best:
        best = mid
        best_pt = mp
    for _ in range(steps):
        if np.sign(mid) == np.sign(upper):
            ub = mp
            upper = mid
        else:
            lb = mp
        mp = (ub + lb) / 2
        mid = func(mp)
        if mid < 0 and mid > best:
            best = mid
            best_pt = mp
    return (best_pt, best)

def proportion_confint(count, nobs, alpha: float=0.05, method='normal'):
    if False:
        return 10
    '\n    Confidence interval for a binomial proportion\n\n    Parameters\n    ----------\n    count : {int or float, array_like}\n        number of successes, can be pandas Series or DataFrame. Arrays\n        must contain integer values if method is "binom_test".\n    nobs : {int or float, array_like}\n        total number of trials.  Arrays must contain integer values if method\n        is "binom_test".\n    alpha : float\n        Significance level, default 0.05. Must be in (0, 1)\n    method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}\n        default: "normal"\n        method to use for confidence interval. Supported methods:\n\n         - `normal` : asymptotic normal approximation\n         - `agresti_coull` : Agresti-Coull interval\n         - `beta` : Clopper-Pearson interval based on Beta distribution\n         - `wilson` : Wilson Score interval\n         - `jeffreys` : Jeffreys Bayesian Interval\n         - `binom_test` : Numerical inversion of binom_test\n\n    Returns\n    -------\n    ci_low, ci_upp : {float, ndarray, Series DataFrame}\n        lower and upper confidence level with coverage (approximately) 1-alpha.\n        When a pandas object is returned, then the index is taken from `count`.\n\n    Notes\n    -----\n    Beta, the Clopper-Pearson exact interval has coverage at least 1-alpha,\n    but is in general conservative. Most of the other methods have average\n    coverage equal to 1-alpha, but will have smaller coverage in some cases.\n\n    The "beta" and "jeffreys" interval are central, they use alpha/2 in each\n    tail, and alpha is not adjusted at the boundaries. In the extreme case\n    when `count` is zero or equal to `nobs`, then the coverage will be only\n    1 - alpha/2 in the case of "beta".\n\n    The confidence intervals are clipped to be in the [0, 1] interval in the\n    case of "normal" and "agresti_coull".\n\n    Method "binom_test" directly inverts the binomial test in scipy.stats.\n    which has discrete steps.\n\n    TODO: binom_test intervals raise an exception in small samples if one\n       interval bound is close to zero or one.\n\n    References\n    ----------\n    .. [*] https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval\n\n    .. [*] Brown, Lawrence D.; Cai, T. Tony; DasGupta, Anirban (2001).\n       "Interval Estimation for a Binomial Proportion", Statistical\n       Science 16 (2): 101–133. doi:10.1214/ss/1009213286.\n    '
    is_scalar = np.isscalar(count) and np.isscalar(nobs)
    is_pandas = isinstance(count, (pd.Series, pd.DataFrame))
    count_a = array_like(count, 'count', optional=False, ndim=None)
    nobs_a = array_like(nobs, 'nobs', optional=False, ndim=None)

    def _check(x: np.ndarray, name: str) -> np.ndarray:
        if False:
            print('Hello World!')
        if np.issubdtype(x.dtype, np.integer):
            return x
        y = x.astype(np.int64, casting='unsafe')
        if np.any(y != x):
            raise ValueError(f'{name} must have an integral dtype. Found data with dtype {x.dtype}')
        return y
    if method == 'binom_test':
        count_a = _check(np.asarray(count_a), 'count')
        nobs_a = _check(np.asarray(nobs_a), 'count')
    q_ = count_a / nobs_a
    alpha_2 = 0.5 * alpha
    if method == 'normal':
        std_ = np.sqrt(q_ * (1 - q_) / nobs_a)
        dist = stats.norm.isf(alpha / 2.0) * std_
        ci_low = q_ - dist
        ci_upp = q_ + dist
    elif method == 'binom_test':

        def func_factory(count: int, nobs: int) -> Callable[[float], float]:
            if False:
                print('Hello World!')
            if hasattr(stats, 'binomtest'):

                def func(qi):
                    if False:
                        print('Hello World!')
                    return stats.binomtest(count, nobs, p=qi).pvalue - alpha
            else:

                def func(qi):
                    if False:
                        print('Hello World!')
                    return stats.binom_test(count, nobs, p=qi) - alpha
            return func
        bcast = np.broadcast(count_a, nobs_a)
        ci_low = np.zeros(bcast.shape)
        ci_upp = np.zeros(bcast.shape)
        index = bcast.index
        for (c, n) in bcast:
            reverse = False
            _q = q_.flat[index]
            if c > n // 2:
                c = n - c
                reverse = True
                _q = 1 - _q
            func = func_factory(c, n)
            if c == 0:
                ci_low.flat[index] = 0.0
            else:
                lower_bnd = _bound_proportion_confint(func, _q, lower=True)
                (val, _z) = optimize.brentq(func, lower_bnd, _q, full_output=True)
                if func(val) > 0:
                    power = 10
                    new_lb = val - (val - lower_bnd) / 2 ** power
                    while func(new_lb) > 0 and power >= 0:
                        power -= 1
                        new_lb = val - (val - lower_bnd) / 2 ** power
                    (val, _) = _bisection_search_conservative(func, new_lb, _q)
                ci_low.flat[index] = val
            if c == n:
                ci_upp.flat[index] = 1.0
            else:
                upper_bnd = _bound_proportion_confint(func, _q, lower=False)
                (val, _z) = optimize.brentq(func, _q, upper_bnd, full_output=True)
                if func(val) > 0:
                    power = 10
                    new_ub = val + (upper_bnd - val) / 2 ** power
                    while func(new_ub) > 0 and power >= 0:
                        power -= 1
                        new_ub = val - (upper_bnd - val) / 2 ** power
                    (val, _) = _bisection_search_conservative(func, _q, new_ub)
                ci_upp.flat[index] = val
            if reverse:
                temp = ci_upp.flat[index]
                ci_upp.flat[index] = 1 - ci_low.flat[index]
                ci_low.flat[index] = 1 - temp
            index = bcast.index
    elif method == 'beta':
        ci_low = stats.beta.ppf(alpha_2, count_a, nobs_a - count_a + 1)
        ci_upp = stats.beta.isf(alpha_2, count_a + 1, nobs_a - count_a)
        if np.ndim(ci_low) > 0:
            ci_low.flat[q_.flat == 0] = 0
            ci_upp.flat[q_.flat == 1] = 1
        else:
            ci_low = 0 if q_ == 0 else ci_low
            ci_upp = 1 if q_ == 1 else ci_upp
    elif method == 'agresti_coull':
        crit = stats.norm.isf(alpha / 2.0)
        nobs_c = nobs_a + crit ** 2
        q_c = (count_a + crit ** 2 / 2.0) / nobs_c
        std_c = np.sqrt(q_c * (1.0 - q_c) / nobs_c)
        dist = crit * std_c
        ci_low = q_c - dist
        ci_upp = q_c + dist
    elif method == 'wilson':
        crit = stats.norm.isf(alpha / 2.0)
        crit2 = crit ** 2
        denom = 1 + crit2 / nobs_a
        center = (q_ + crit2 / (2 * nobs_a)) / denom
        dist = crit * np.sqrt(q_ * (1.0 - q_) / nobs_a + crit2 / (4.0 * nobs_a ** 2))
        dist /= denom
        ci_low = center - dist
        ci_upp = center + dist
    elif method[:4] == 'jeff':
        (ci_low, ci_upp) = stats.beta.interval(1 - alpha, count_a + 0.5, nobs_a - count_a + 0.5)
    else:
        raise NotImplementedError(f'method {method} is not available')
    if method in ['normal', 'agresti_coull']:
        ci_low = np.clip(ci_low, 0, 1)
        ci_upp = np.clip(ci_upp, 0, 1)
    if is_pandas:
        container = pd.Series if isinstance(count, pd.Series) else pd.DataFrame
        ci_low = container(ci_low, index=count.index)
        ci_upp = container(ci_upp, index=count.index)
    if is_scalar:
        return (float(ci_low), float(ci_upp))
    return (ci_low, ci_upp)

def multinomial_proportions_confint(counts, alpha=0.05, method='goodman'):
    if False:
        print('Hello World!')
    '\n    Confidence intervals for multinomial proportions.\n\n    Parameters\n    ----------\n    counts : array_like of int, 1-D\n        Number of observations in each category.\n    alpha : float in (0, 1), optional\n        Significance level, defaults to 0.05.\n    method : {\'goodman\', \'sison-glaz\'}, optional\n        Method to use to compute the confidence intervals; available methods\n        are:\n\n         - `goodman`: based on a chi-squared approximation, valid if all\n           values in `counts` are greater or equal to 5 [2]_\n         - `sison-glaz`: less conservative than `goodman`, but only valid if\n           `counts` has 7 or more categories (``len(counts) >= 7``) [3]_\n\n    Returns\n    -------\n    confint : ndarray, 2-D\n        Array of [lower, upper] confidence levels for each category, such that\n        overall coverage is (approximately) `1-alpha`.\n\n    Raises\n    ------\n    ValueError\n        If `alpha` is not in `(0, 1)` (bounds excluded), or if the values in\n        `counts` are not all positive or null.\n    NotImplementedError\n        If `method` is not kown.\n    Exception\n        When ``method == \'sison-glaz\'``, if for some reason `c` cannot be\n        computed; this signals a bug and should be reported.\n\n    Notes\n    -----\n    The `goodman` method [2]_ is based on approximating a statistic based on\n    the multinomial as a chi-squared random variable. The usual recommendation\n    is that this is valid if all the values in `counts` are greater than or\n    equal to 5. There is no condition on the number of categories for this\n    method.\n\n    The `sison-glaz` method [3]_ approximates the multinomial probabilities,\n    and evaluates that with a maximum-likelihood estimator. The first\n    approximation is an Edgeworth expansion that converges when the number of\n    categories goes to infinity, and the maximum-likelihood estimator converges\n    when the number of observations (``sum(counts)``) goes to infinity. In\n    their paper, Sison & Glaz demo their method with at least 7 categories, so\n    ``len(counts) >= 7`` with all values in `counts` at or above 5 can be used\n    as a rule of thumb for the validity of this method. This method is less\n    conservative than the `goodman` method (i.e. it will yield confidence\n    intervals closer to the desired significance level), but produces\n    confidence intervals of uniform width over all categories (except when the\n    intervals reach 0 or 1, in which case they are truncated), which makes it\n    most useful when proportions are of similar magnitude.\n\n    Aside from the original sources ([1]_, [2]_, and [3]_), the implementation\n    uses the formulas (though not the code) presented in [4]_ and [5]_.\n\n    References\n    ----------\n    .. [1] Levin, Bruce, "A representation for multinomial cumulative\n           distribution functions," The Annals of Statistics, Vol. 9, No. 5,\n           1981, pp. 1123-1126.\n\n    .. [2] Goodman, L.A., "On simultaneous confidence intervals for multinomial\n           proportions," Technometrics, Vol. 7, No. 2, 1965, pp. 247-254.\n\n    .. [3] Sison, Cristina P., and Joseph Glaz, "Simultaneous Confidence\n           Intervals and Sample Size Determination for Multinomial\n           Proportions," Journal of the American Statistical Association,\n           Vol. 90, No. 429, 1995, pp. 366-369.\n\n    .. [4] May, Warren L., and William D. Johnson, "A SAS® macro for\n           constructing simultaneous confidence intervals  for multinomial\n           proportions," Computer methods and programs in Biomedicine, Vol. 53,\n           No. 3, 1997, pp. 153-162.\n\n    .. [5] May, Warren L., and William D. Johnson, "Constructing two-sided\n           simultaneous confidence intervals for multinomial proportions for\n           small counts in a large number of cells," Journal of Statistical\n           Software, Vol. 5, No. 6, 2000, pp. 1-24.\n    '
    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in (0, 1), bounds excluded')
    counts = np.array(counts, dtype=float)
    if (counts < 0).any():
        raise ValueError('counts must be >= 0')
    n = counts.sum()
    k = len(counts)
    proportions = counts / n
    if method == 'goodman':
        chi2 = stats.chi2.ppf(1 - alpha / k, 1)
        delta = chi2 ** 2 + 4 * n * proportions * chi2 * (1 - proportions)
        region = ((2 * n * proportions + chi2 + np.array([-np.sqrt(delta), np.sqrt(delta)])) / (2 * (chi2 + n))).T
    elif method[:5] == 'sison':

        def poisson_interval(interval, p):
            if False:
                while True:
                    i = 10
            '\n            Compute P(b <= Z <= a) where Z ~ Poisson(p) and\n            `interval = (b, a)`.\n            '
            (b, a) = interval
            prob = stats.poisson.cdf(a, p) - stats.poisson.cdf(b - 1, p)
            return prob

        def truncated_poisson_factorial_moment(interval, r, p):
            if False:
                i = 10
                return i + 15
            '\n            Compute mu_r, the r-th factorial moment of a poisson random\n            variable of parameter `p` truncated to `interval = (b, a)`.\n            '
            (b, a) = interval
            return p ** r * (1 - (poisson_interval((a - r + 1, a), p) - poisson_interval((b - r, b - 1), p)) / poisson_interval((b, a), p))

        def edgeworth(intervals):
            if False:
                return 10
            "\n            Compute the Edgeworth expansion term of Sison & Glaz's formula\n            (1) (approximated probability for multinomial proportions in a\n            given box).\n            "
            (mu_r1, mu_r2, mu_r3, mu_r4) = [np.array([truncated_poisson_factorial_moment(interval, r, p) for (interval, p) in zip(intervals, counts)]) for r in range(1, 5)]
            mu = mu_r1
            mu2 = mu_r2 + mu - mu ** 2
            mu3 = mu_r3 + mu_r2 * (3 - 3 * mu) + mu - 3 * mu ** 2 + 2 * mu ** 3
            mu4 = mu_r4 + mu_r3 * (6 - 4 * mu) + mu_r2 * (7 - 12 * mu + 6 * mu ** 2) + mu - 4 * mu ** 2 + 6 * mu ** 3 - 3 * mu ** 4
            g1 = mu3.sum() / mu2.sum() ** 1.5
            g2 = (mu4.sum() - 3 * (mu2 ** 2).sum()) / mu2.sum() ** 2
            x = (n - mu.sum()) / np.sqrt(mu2.sum())
            phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
            H3 = x ** 3 - 3 * x
            H4 = x ** 4 - 6 * x ** 2 + 3
            H6 = x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15
            f = phi * (1 + g1 * H3 / 6 + g2 * H4 / 24 + g1 ** 2 * H6 / 72)
            return f / np.sqrt(mu2.sum())

        def approximated_multinomial_interval(intervals):
            if False:
                return 10
            "\n            Compute approximated probability for Multinomial(n, proportions)\n            to be in `intervals` (Sison & Glaz's formula (1)).\n            "
            return np.exp(np.sum(np.log([poisson_interval(interval, p) for (interval, p) in zip(intervals, counts)])) + np.log(edgeworth(intervals)) - np.log(stats.poisson._pmf(n, n)))

        def nu(c):
            if False:
                return 10
            "\n            Compute interval coverage for a given `c` (Sison & Glaz's\n            formula (7)).\n            "
            return approximated_multinomial_interval([(np.maximum(count - c, 0), np.minimum(count + c, n)) for count in counts])
        c = 1.0
        nuc = nu(c)
        nucp1 = nu(c + 1)
        while not nuc <= 1 - alpha < nucp1:
            if c > n:
                raise Exception("Couldn't find a value for `c` that solves nu(c) <= 1 - alpha < nu(c + 1)")
            c += 1
            nuc = nucp1
            nucp1 = nu(c + 1)
        g = (1 - alpha - nuc) / (nucp1 - nuc)
        ci_lower = np.maximum(proportions - c / n, 0)
        ci_upper = np.minimum(proportions + (c + 2 * g) / n, 1)
        region = np.array([ci_lower, ci_upper]).T
    else:
        raise NotImplementedError('method "%s" is not available' % method)
    return region

def samplesize_confint_proportion(proportion, half_length, alpha=0.05, method='normal'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find sample size to get desired confidence interval length\n\n    Parameters\n    ----------\n    proportion : float in (0, 1)\n        proportion or quantile\n    half_length : float in (0, 1)\n        desired half length of the confidence interval\n    alpha : float in (0, 1)\n        significance level, default 0.05,\n        coverage of the two-sided interval is (approximately) ``1 - alpha``\n    method : str in ['normal']\n        method to use for confidence interval,\n        currently only normal approximation\n\n    Returns\n    -------\n    n : float\n        sample size to get the desired half length of the confidence interval\n\n    Notes\n    -----\n    this is mainly to store the formula.\n    possible application: number of replications in bootstrap samples\n\n    "
    q_ = proportion
    if method == 'normal':
        n = q_ * (1 - q_) / (half_length / stats.norm.isf(alpha / 2.0)) ** 2
    else:
        raise NotImplementedError('only "normal" is available')
    return n

def proportion_effectsize(prop1, prop2, method='normal'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Effect size for a test comparing two proportions\n\n    for use in power function\n\n    Parameters\n    ----------\n    prop1, prop2 : float or array_like\n        The proportion value(s).\n\n    Returns\n    -------\n    es : float or ndarray\n        effect size for (transformed) prop1 - prop2\n\n    Notes\n    -----\n    only method='normal' is implemented to match pwr.p2.test\n    see http://www.statmethods.net/stats/power.html\n\n    Effect size for `normal` is defined as ::\n\n        2 * (arcsin(sqrt(prop1)) - arcsin(sqrt(prop2)))\n\n    I think other conversions to normality can be used, but I need to check.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> sm.stats.proportion_effectsize(0.5, 0.4)\n    0.20135792079033088\n    >>> sm.stats.proportion_effectsize([0.3, 0.4, 0.5], 0.4)\n    array([-0.21015893,  0.        ,  0.20135792])\n\n    "
    if method != 'normal':
        raise ValueError('only "normal" is implemented')
    es = 2 * (np.arcsin(np.sqrt(prop1)) - np.arcsin(np.sqrt(prop2)))
    return es

def std_prop(prop, nobs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Standard error for the estimate of a proportion\n\n    This is just ``np.sqrt(p * (1. - p) / nobs)``\n\n    Parameters\n    ----------\n    prop : array_like\n        proportion\n    nobs : int, array_like\n        number of observations\n\n    Returns\n    -------\n    std : array_like\n        standard error for a proportion of nobs independent observations\n    '
    return np.sqrt(prop * (1.0 - prop) / nobs)

def _std_diff_prop(p1, p2, ratio=1):
    if False:
        while True:
            i = 10
    return np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)

def _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt, alpha=0.05, discrete=True, dist='norm', nobs=None, continuity=0, critval_continuity=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generic statistical power function for normal based equivalence test\n\n    This includes options to adjust the normal approximation and can use\n    the binomial to evaluate the probability of the rejection region\n\n    see power_ztost_prob for a description of the options\n    '
    if not isinstance(continuity, tuple):
        continuity = (continuity, continuity)
    crit = stats.norm.isf(alpha)
    k_low = mean_low + np.sqrt(var_low) * crit
    k_upp = mean_upp - np.sqrt(var_upp) * crit
    if discrete or dist == 'binom':
        k_low = np.ceil(k_low * nobs + 0.5 * critval_continuity)
        k_upp = np.trunc(k_upp * nobs - 0.5 * critval_continuity)
        if dist == 'norm':
            k_low = k_low * 1.0 / nobs
            k_upp = k_upp * 1.0 / nobs
    if np.any(k_low > k_upp):
        import warnings
        warnings.warn('no overlap, power is zero', HypothesisTestWarning)
    std_alt = np.sqrt(var_alt)
    z_low = (k_low - mean_alt - continuity[0] * 0.5 / nobs) / std_alt
    z_upp = (k_upp - mean_alt + continuity[1] * 0.5 / nobs) / std_alt
    if dist == 'norm':
        power = stats.norm.cdf(z_upp) - stats.norm.cdf(z_low)
    elif dist == 'binom':
        power = stats.binom.cdf(k_upp, nobs, mean_alt) - stats.binom.cdf(k_low - 1, nobs, mean_alt)
    return (power, (k_low, k_upp, z_low, z_upp))

def binom_tost(count, nobs, low, upp):
    if False:
        i = 10
        return i + 15
    '\n    Exact TOST test for one proportion using binomial distribution\n\n    Parameters\n    ----------\n    count : {int, array_like}\n        the number of successes in nobs trials.\n    nobs : int\n        the number of trials or observations.\n    low, upp : floats\n        lower and upper limit of equivalence region\n\n    Returns\n    -------\n    pvalue : float\n        p-value of equivalence test\n    pval_low, pval_upp : floats\n        p-values of lower and upper one-sided tests\n\n    '
    tt1 = binom_test(count, nobs, alternative='larger', prop=low)
    tt2 = binom_test(count, nobs, alternative='smaller', prop=upp)
    return (np.maximum(tt1, tt2), tt1, tt2)

def binom_tost_reject_interval(low, upp, nobs, alpha=0.05):
    if False:
        print('Hello World!')
    '\n    Rejection region for binomial TOST\n\n    The interval includes the end points,\n    `reject` if and only if `r_low <= x <= r_upp`.\n\n    The interval might be empty with `r_upp < r_low`.\n\n    Parameters\n    ----------\n    low, upp : floats\n        lower and upper limit of equivalence region\n    nobs : int\n        the number of trials or observations.\n\n    Returns\n    -------\n    x_low, x_upp : float\n        lower and upper bound of rejection region\n\n    '
    x_low = stats.binom.isf(alpha, nobs, low) + 1
    x_upp = stats.binom.ppf(alpha, nobs, upp) - 1
    return (x_low, x_upp)

def binom_test_reject_interval(value, nobs, alpha=0.05, alternative='two-sided'):
    if False:
        print('Hello World!')
    '\n    Rejection region for binomial test for one sample proportion\n\n    The interval includes the end points of the rejection region.\n\n    Parameters\n    ----------\n    value : float\n        proportion under the Null hypothesis\n    nobs : int\n        the number of trials or observations.\n\n    Returns\n    -------\n    x_low, x_upp : int\n        lower and upper bound of rejection region\n    '
    if alternative in ['2s', 'two-sided']:
        alternative = '2s'
        alpha = alpha / 2
    if alternative in ['2s', 'smaller']:
        x_low = stats.binom.ppf(alpha, nobs, value) - 1
    else:
        x_low = 0
    if alternative in ['2s', 'larger']:
        x_upp = stats.binom.isf(alpha, nobs, value) + 1
    else:
        x_upp = nobs
    return (int(x_low), int(x_upp))

def binom_test(count, nobs, prop=0.5, alternative='two-sided'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Perform a test that the probability of success is p.\n\n    This is an exact, two-sided test of the null hypothesis\n    that the probability of success in a Bernoulli experiment\n    is `p`.\n\n    Parameters\n    ----------\n    count : {int, array_like}\n        the number of successes in nobs trials.\n    nobs : int\n        the number of trials or observations.\n    prop : float, optional\n        The probability of success under the null hypothesis,\n        `0 <= prop <= 1`. The default value is `prop = 0.5`\n    alternative : str in ['two-sided', 'smaller', 'larger']\n        alternative hypothesis, which can be two-sided or either one of the\n        one-sided tests.\n\n    Returns\n    -------\n    p-value : float\n        The p-value of the hypothesis test\n\n    Notes\n    -----\n    This uses scipy.stats.binom_test for the two-sided alternative.\n    "
    if np.any(prop > 1.0) or np.any(prop < 0.0):
        raise ValueError('p must be in range [0,1]')
    if alternative in ['2s', 'two-sided']:
        try:
            pval = stats.binomtest(count, n=nobs, p=prop).pvalue
        except AttributeError:
            pval = stats.binom_test(count, n=nobs, p=prop)
    elif alternative in ['l', 'larger']:
        pval = stats.binom.sf(count - 1, nobs, prop)
    elif alternative in ['s', 'smaller']:
        pval = stats.binom.cdf(count, nobs, prop)
    else:
        raise ValueError('alternative not recognized\nshould be two-sided, larger or smaller')
    return pval

def power_binom_tost(low, upp, nobs, p_alt=None, alpha=0.05):
    if False:
        return 10
    if p_alt is None:
        p_alt = 0.5 * (low + upp)
    (x_low, x_upp) = binom_tost_reject_interval(low, upp, nobs, alpha=alpha)
    power = stats.binom.cdf(x_upp, nobs, p_alt) - stats.binom.cdf(x_low - 1, nobs, p_alt)
    return power

def power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='norm', variance_prop=None, discrete=True, continuity=0, critval_continuity=0):
    if False:
        print('Hello World!')
    '\n    Power of proportions equivalence test based on normal distribution\n\n    Parameters\n    ----------\n    low, upp : floats\n        lower and upper limit of equivalence region\n    nobs : int\n        number of observations\n    p_alt : float in (0,1)\n        proportion under the alternative\n    alpha : float in (0,1)\n        significance level of the test\n    dist : str in [\'norm\', \'binom\']\n        This defines the distribution to evaluate the power of the test. The\n        critical values of the TOST test are always based on the normal\n        approximation, but the distribution for the power can be either the\n        normal (default) or the binomial (exact) distribution.\n    variance_prop : None or float in (0,1)\n        If this is None, then the variances for the two one sided tests are\n        based on the proportions equal to the equivalence limits.\n        If variance_prop is given, then it is used to calculate the variance\n        for the TOST statistics. If this is based on an sample, then the\n        estimated proportion can be used.\n    discrete : bool\n        If true, then the critical values of the rejection region are converted\n        to integers. If dist is "binom", this is automatically assumed.\n        If discrete is false, then the TOST critical values are used as\n        floating point numbers, and the power is calculated based on the\n        rejection region that is not discretized.\n    continuity : bool or float\n        adjust the rejection region for the normal power probability. This has\n        and effect only if ``dist=\'norm\'``\n    critval_continuity : bool or float\n        If this is non-zero, then the critical values of the tost rejection\n        region are adjusted before converting to integers. This affects both\n        distributions, ``dist=\'norm\'`` and ``dist=\'binom\'``.\n\n    Returns\n    -------\n    power : float\n        statistical power of the equivalence test.\n    (k_low, k_upp, z_low, z_upp) : tuple of floats\n        critical limits in intermediate steps\n        temporary return, will be changed\n\n    Notes\n    -----\n    In small samples the power for the ``discrete`` version, has a sawtooth\n    pattern as a function of the number of observations. As a consequence,\n    small changes in the number of observations or in the normal approximation\n    can have a large effect on the power.\n\n    ``continuity`` and ``critval_continuity`` are added to match some results\n    of PASS, and are mainly to investigate the sensitivity of the ztost power\n    to small changes in the rejection region. From my interpretation of the\n    equations in the SAS manual, both are zero in SAS.\n\n    works vectorized\n\n    **verification:**\n\n    The ``dist=\'binom\'`` results match PASS,\n    The ``dist=\'norm\'`` results look reasonable, but no benchmark is available.\n\n    References\n    ----------\n    SAS Manual: Chapter 68: The Power Procedure, Computational Resources\n    PASS Chapter 110: Equivalence Tests for One Proportion.\n\n    '
    mean_low = low
    var_low = std_prop(low, nobs) ** 2
    mean_upp = upp
    var_upp = std_prop(upp, nobs) ** 2
    mean_alt = p_alt
    var_alt = std_prop(p_alt, nobs) ** 2
    if variance_prop is not None:
        var_low = var_upp = std_prop(variance_prop, nobs) ** 2
    power = _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt, alpha=alpha, discrete=discrete, dist=dist, nobs=nobs, continuity=continuity, critval_continuity=critval_continuity)
    return (np.maximum(power[0], 0), power[1:])

def _table_proportion(count, nobs):
    if False:
        i = 10
        return i + 15
    '\n    Create a k by 2 contingency table for proportion\n\n    helper function for proportions_chisquare\n\n    Parameters\n    ----------\n    count : {int, array_like}\n        the number of successes in nobs trials.\n    nobs : int\n        the number of trials or observations.\n\n    Returns\n    -------\n    table : ndarray\n        (k, 2) contingency table\n\n    Notes\n    -----\n    recent scipy has more elaborate contingency table functions\n\n    '
    count = np.asarray(count)
    dt = np.promote_types(count.dtype, np.float64)
    count = np.asarray(count, dtype=dt)
    table = np.column_stack((count, nobs - count))
    expected = table.sum(0) * table.sum(1)[:, None] * 1.0 / table.sum()
    n_rows = table.shape[0]
    return (table, expected, n_rows)

def proportions_ztest(count, nobs, value=None, alternative='two-sided', prop_var=False):
    if False:
        return 10
    "\n    Test for proportions based on normal (z) test\n\n    Parameters\n    ----------\n    count : {int, array_like}\n        the number of successes in nobs trials. If this is array_like, then\n        the assumption is that this represents the number of successes for\n        each independent sample\n    nobs : {int, array_like}\n        the number of trials or observations, with the same length as\n        count.\n    value : float, array_like or None, optional\n        This is the value of the null hypothesis equal to the proportion in the\n        case of a one sample test. In the case of a two-sample test, the\n        null hypothesis is that prop[0] - prop[1] = value, where prop is the\n        proportion in the two samples. If not provided value = 0 and the null\n        is prop[0] = prop[1]\n    alternative : str in ['two-sided', 'smaller', 'larger']\n        The alternative hypothesis can be either two-sided or one of the one-\n        sided tests, smaller means that the alternative hypothesis is\n        ``prop < value`` and larger means ``prop > value``. In the two sample\n        test, smaller means that the alternative hypothesis is ``p1 < p2`` and\n        larger means ``p1 > p2`` where ``p1`` is the proportion of the first\n        sample and ``p2`` of the second one.\n    prop_var : False or float in (0, 1)\n        If prop_var is false, then the variance of the proportion estimate is\n        calculated based on the sample proportion. Alternatively, a proportion\n        can be specified to calculate this variance. Common use case is to\n        use the proportion under the Null hypothesis to specify the variance\n        of the proportion estimate.\n\n    Returns\n    -------\n    zstat : float\n        test statistic for the z-test\n    p-value : float\n        p-value for the z-test\n\n    Examples\n    --------\n    >>> count = 5\n    >>> nobs = 83\n    >>> value = .05\n    >>> stat, pval = proportions_ztest(count, nobs, value)\n    >>> print('{0:0.3f}'.format(pval))\n    0.695\n\n    >>> import numpy as np\n    >>> from statsmodels.stats.proportion import proportions_ztest\n    >>> count = np.array([5, 12])\n    >>> nobs = np.array([83, 99])\n    >>> stat, pval = proportions_ztest(count, nobs)\n    >>> print('{0:0.3f}'.format(pval))\n    0.159\n\n    Notes\n    -----\n    This uses a simple normal test for proportions. It should be the same as\n    running the mean z-test on the data encoded 1 for event and 0 for no event\n    so that the sum corresponds to the count.\n\n    In the one and two sample cases with two-sided alternative, this test\n    produces the same p-value as ``proportions_chisquare``, since the\n    chisquare is the distribution of the square of a standard normal\n    distribution.\n    "
    count = np.asarray(count)
    nobs = np.asarray(nobs)
    if nobs.size == 1:
        nobs = nobs * np.ones_like(count)
    prop = count * 1.0 / nobs
    k_sample = np.size(prop)
    if value is None:
        if k_sample == 1:
            raise ValueError('value must be provided for a 1-sample test')
        value = 0
    if k_sample == 1:
        diff = prop - value
    elif k_sample == 2:
        diff = prop[0] - prop[1] - value
    else:
        msg = 'more than two samples are not implemented yet'
        raise NotImplementedError(msg)
    p_pooled = np.sum(count) * 1.0 / np.sum(nobs)
    nobs_fact = np.sum(1.0 / nobs)
    if prop_var:
        p_pooled = prop_var
    var_ = p_pooled * (1 - p_pooled) * nobs_fact
    std_diff = np.sqrt(var_)
    from statsmodels.stats.weightstats import _zstat_generic2
    return _zstat_generic2(diff, std_diff, alternative)

def proportions_ztost(count, nobs, low, upp, prop_var='sample'):
    if False:
        print('Hello World!')
    "\n    Equivalence test based on normal distribution\n\n    Parameters\n    ----------\n    count : {int, array_like}\n        the number of successes in nobs trials. If this is array_like, then\n        the assumption is that this represents the number of successes for\n        each independent sample\n    nobs : int\n        the number of trials or observations, with the same length as\n        count.\n    low, upp : float\n        equivalence interval low < prop1 - prop2 < upp\n    prop_var : str or float in (0, 1)\n        prop_var determines which proportion is used for the calculation\n        of the standard deviation of the proportion estimate\n        The available options for string are 'sample' (default), 'null' and\n        'limits'. If prop_var is a float, then it is used directly.\n\n    Returns\n    -------\n    pvalue : float\n        pvalue of the non-equivalence test\n    t1, pv1 : tuple of floats\n        test statistic and pvalue for lower threshold test\n    t2, pv2 : tuple of floats\n        test statistic and pvalue for upper threshold test\n\n    Notes\n    -----\n    checked only for 1 sample case\n\n    "
    if prop_var == 'limits':
        prop_var_low = low
        prop_var_upp = upp
    elif prop_var == 'sample':
        prop_var_low = prop_var_upp = False
    elif prop_var == 'null':
        prop_var_low = prop_var_upp = 0.5 * (low + upp)
    elif np.isreal(prop_var):
        prop_var_low = prop_var_upp = prop_var
    tt1 = proportions_ztest(count, nobs, alternative='larger', prop_var=prop_var_low, value=low)
    tt2 = proportions_ztest(count, nobs, alternative='smaller', prop_var=prop_var_upp, value=upp)
    return (np.maximum(tt1[1], tt2[1]), tt1, tt2)

def proportions_chisquare(count, nobs, value=None):
    if False:
        return 10
    '\n    Test for proportions based on chisquare test\n\n    Parameters\n    ----------\n    count : {int, array_like}\n        the number of successes in nobs trials. If this is array_like, then\n        the assumption is that this represents the number of successes for\n        each independent sample\n    nobs : int\n        the number of trials or observations, with the same length as\n        count.\n    value : None or float or array_like\n\n    Returns\n    -------\n    chi2stat : float\n        test statistic for the chisquare test\n    p-value : float\n        p-value for the chisquare test\n    (table, expected)\n        table is a (k, 2) contingency table, ``expected`` is the corresponding\n        table of counts that are expected under independence with given\n        margins\n\n    Notes\n    -----\n    Recent version of scipy.stats have a chisquare test for independence in\n    contingency tables.\n\n    This function provides a similar interface to chisquare tests as\n    ``prop.test`` in R, however without the option for Yates continuity\n    correction.\n\n    count can be the count for the number of events for a single proportion,\n    or the counts for several independent proportions. If value is given, then\n    all proportions are jointly tested against this value. If value is not\n    given and count and nobs are not scalar, then the null hypothesis is\n    that all samples have the same proportion.\n\n    '
    nobs = np.atleast_1d(nobs)
    (table, expected, n_rows) = _table_proportion(count, nobs)
    if value is not None:
        expected = np.column_stack((nobs * value, nobs * (1 - value)))
        ddof = n_rows - 1
    else:
        ddof = n_rows
    (chi2stat, pval) = stats.chisquare(table.ravel(), expected.ravel(), ddof=ddof)
    return (chi2stat, pval, (table, expected))

def proportions_chisquare_allpairs(count, nobs, multitest_method='hs'):
    if False:
        print('Hello World!')
    "\n    Chisquare test of proportions for all pairs of k samples\n\n    Performs a chisquare test for proportions for all pairwise comparisons.\n    The alternative is two-sided\n\n    Parameters\n    ----------\n    count : {int, array_like}\n        the number of successes in nobs trials.\n    nobs : int\n        the number of trials or observations.\n    multitest_method : str\n        This chooses the method for the multiple testing p-value correction,\n        that is used as default in the results.\n        It can be any method that is available in  ``multipletesting``.\n        The default is Holm-Sidak 'hs'.\n\n    Returns\n    -------\n    result : AllPairsResults instance\n        The returned results instance has several statistics, such as p-values,\n        attached, and additional methods for using a non-default\n        ``multitest_method``.\n\n    Notes\n    -----\n    Yates continuity correction is not available.\n    "
    all_pairs = lzip(*np.triu_indices(len(count), 1))
    pvals = [proportions_chisquare(count[list(pair)], nobs[list(pair)])[1] for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)

def proportions_chisquare_pairscontrol(count, nobs, value=None, multitest_method='hs', alternative='two-sided'):
    if False:
        return 10
    "\n    Chisquare test of proportions for pairs of k samples compared to control\n\n    Performs a chisquare test for proportions for pairwise comparisons with a\n    control (Dunnet's test). The control is assumed to be the first element\n    of ``count`` and ``nobs``. The alternative is two-sided, larger or\n    smaller.\n\n    Parameters\n    ----------\n    count : {int, array_like}\n        the number of successes in nobs trials.\n    nobs : int\n        the number of trials or observations.\n    multitest_method : str\n        This chooses the method for the multiple testing p-value correction,\n        that is used as default in the results.\n        It can be any method that is available in  ``multipletesting``.\n        The default is Holm-Sidak 'hs'.\n    alternative : str in ['two-sided', 'smaller', 'larger']\n        alternative hypothesis, which can be two-sided or either one of the\n        one-sided tests.\n\n    Returns\n    -------\n    result : AllPairsResults instance\n        The returned results instance has several statistics, such as p-values,\n        attached, and additional methods for using a non-default\n        ``multitest_method``.\n\n\n    Notes\n    -----\n    Yates continuity correction is not available.\n\n    ``value`` and ``alternative`` options are not yet implemented.\n\n    "
    if value is not None or alternative not in ['two-sided', '2s']:
        raise NotImplementedError
    all_pairs = [(0, k) for k in range(1, len(count))]
    pvals = [proportions_chisquare(count[list(pair)], nobs[list(pair)])[1] for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)

def confint_proportions_2indep(count1, nobs1, count2, nobs2, method=None, compare='diff', alpha=0.05, correction=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Confidence intervals for comparing two independent proportions.\n\n    This assumes that we have two independent binomial samples.\n\n    Parameters\n    ----------\n    count1, nobs1 : float\n        Count and sample size for first sample.\n    count2, nobs2 : float\n        Count and sample size for the second sample.\n    method : str\n        Method for computing confidence interval. If method is None, then a\n        default method is used. The default might change as more methods are\n        added.\n\n        diff:\n         - \'wald\',\n         - \'agresti-caffo\'\n         - \'newcomb\' (default)\n         - \'score\'\n\n        ratio:\n         - \'log\'\n         - \'log-adjusted\' (default)\n         - \'score\'\n\n        odds-ratio:\n         - \'logit\'\n         - \'logit-adjusted\' (default)\n         - \'score\'\n\n    compare : string in [\'diff\', \'ratio\' \'odds-ratio\']\n        If compare is diff, then the confidence interval is for diff = p1 - p2.\n        If compare is ratio, then the confidence interval is for the risk ratio\n        defined by ratio = p1 / p2.\n        If compare is odds-ratio, then the confidence interval is for the\n        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).\n    alpha : float\n        Significance level for the confidence interval, default is 0.05.\n        The nominal coverage probability is 1 - alpha.\n\n    Returns\n    -------\n    low, upp\n\n    See Also\n    --------\n    test_proportions_2indep\n    tost_proportions_2indep\n\n    Notes\n    -----\n    Status: experimental, API and defaults might still change.\n        more ``methods`` will be added.\n\n    References\n    ----------\n    .. [1] Fagerland, Morten W., Stian Lydersen, and Petter Laake. 2015.\n       “Recommended Confidence Intervals for Two Independent Binomial\n       Proportions.” Statistical Methods in Medical Research 24 (2): 224–54.\n       https://doi.org/10.1177/0962280211415469.\n    .. [2] Koopman, P. A. R. 1984. “Confidence Intervals for the Ratio of Two\n       Binomial Proportions.” Biometrics 40 (2): 513–17.\n       https://doi.org/10.2307/2531405.\n    .. [3] Miettinen, Olli, and Markku Nurminen. "Comparative analysis of two\n       rates." Statistics in medicine 4, no. 2 (1985): 213-226.\n    .. [4] Newcombe, Robert G. 1998. “Interval Estimation for the Difference\n       between Independent Proportions: Comparison of Eleven Methods.”\n       Statistics in Medicine 17 (8): 873–90.\n       https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8<873::AID-\n       SIM779>3.0.CO;2-I.\n    .. [5] Newcombe, Robert G., and Markku M. Nurminen. 2011. “In Defence of\n       Score Intervals for Proportions and Their Differences.” Communications\n       in Statistics - Theory and Methods 40 (7): 1271–82.\n       https://doi.org/10.1080/03610920903576580.\n    '
    method_default = {'diff': 'newcomb', 'ratio': 'log-adjusted', 'odds-ratio': 'logit-adjusted'}
    if compare.lower() == 'or':
        compare = 'odds-ratio'
    if method is None:
        method = method_default[compare]
    method = method.lower()
    if method.startswith('agr'):
        method = 'agresti-caffo'
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    diff = p1 - p2
    addone = 1 if method == 'agresti-caffo' else 0
    if compare == 'diff':
        if method in ['wald', 'agresti-caffo']:
            (count1_, nobs1_) = (count1 + addone, nobs1 + 2 * addone)
            (count2_, nobs2_) = (count2 + addone, nobs2 + 2 * addone)
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            diff_ = p1_ - p2_
            var = p1_ * (1 - p1_) / nobs1_ + p2_ * (1 - p2_) / nobs2_
            z = stats.norm.isf(alpha / 2)
            d_wald = z * np.sqrt(var)
            low = diff_ - d_wald
            upp = diff_ + d_wald
        elif method.startswith('newcomb'):
            (low1, upp1) = proportion_confint(count1, nobs1, method='wilson', alpha=alpha)
            (low2, upp2) = proportion_confint(count2, nobs2, method='wilson', alpha=alpha)
            d_low = np.sqrt((p1 - low1) ** 2 + (upp2 - p2) ** 2)
            d_upp = np.sqrt((p2 - low2) ** 2 + (upp1 - p1) ** 2)
            low = diff - d_low
            upp = diff + d_upp
        elif method == 'score':
            (low, upp) = _score_confint_inversion(count1, nobs1, count2, nobs2, compare=compare, alpha=alpha, correction=correction)
        else:
            raise ValueError('method not recognized')
    elif compare == 'ratio':
        if method in ['log', 'log-adjusted']:
            addhalf = 0.5 if method == 'log-adjusted' else 0
            (count1_, nobs1_) = (count1 + addhalf, nobs1 + addhalf)
            (count2_, nobs2_) = (count2 + addhalf, nobs2 + addhalf)
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            ratio_ = p1_ / p2_
            var = 1 / count1_ - 1 / nobs1_ + 1 / count2_ - 1 / nobs2_
            z = stats.norm.isf(alpha / 2)
            d_log = z * np.sqrt(var)
            low = np.exp(np.log(ratio_) - d_log)
            upp = np.exp(np.log(ratio_) + d_log)
        elif method == 'score':
            res = _confint_riskratio_koopman(count1, nobs1, count2, nobs2, alpha=alpha, correction=correction)
            (low, upp) = res.confint
        else:
            raise ValueError('method not recognized')
    elif compare == 'odds-ratio':
        if method in ['logit', 'logit-adjusted', 'logit-smoothed']:
            if method in ['logit-smoothed']:
                adjusted = _shrink_prob(count1, nobs1, count2, nobs2, shrink_factor=2, return_corr=False)[0]
                (count1_, nobs1_, count2_, nobs2_) = adjusted
            else:
                addhalf = 0.5 if method == 'logit-adjusted' else 0
                (count1_, nobs1_) = (count1 + addhalf, nobs1 + 2 * addhalf)
                (count2_, nobs2_) = (count2 + addhalf, nobs2 + 2 * addhalf)
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            odds_ratio_ = p1_ / (1 - p1_) / p2_ * (1 - p2_)
            var = 1 / count1_ + 1 / (nobs1_ - count1_) + 1 / count2_ + 1 / (nobs2_ - count2_)
            z = stats.norm.isf(alpha / 2)
            d_log = z * np.sqrt(var)
            low = np.exp(np.log(odds_ratio_) - d_log)
            upp = np.exp(np.log(odds_ratio_) + d_log)
        elif method == 'score':
            (low, upp) = _score_confint_inversion(count1, nobs1, count2, nobs2, compare=compare, alpha=alpha, correction=correction)
        else:
            raise ValueError('method not recognized')
    else:
        raise ValueError('compare not recognized')
    return (low, upp)

def _shrink_prob(count1, nobs1, count2, nobs2, shrink_factor=2, return_corr=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Shrink observed counts towards independence\n\n    Helper function for 'logit-smoothed' inference for the odds-ratio of two\n    independent proportions.\n\n    Parameters\n    ----------\n    count1, nobs1 : float or int\n        count and sample size for first sample\n    count2, nobs2 : float or int\n        count and sample size for the second sample\n    shrink_factor : float\n        This corresponds to the number of observations that are added in total\n        proportional to the probabilities under independence.\n    return_corr : bool\n        If true, then only the correction term is returned\n        If false, then the corrected counts, i.e. original counts plus\n        correction term, are returned.\n\n    Returns\n    -------\n    count1_corr, nobs1_corr, count2_corr, nobs2_corr : float\n        correction or corrected counts\n    prob_indep :\n        TODO/Warning : this will change most likely\n        probabilities under independence, only returned if return_corr is\n        false.\n\n    "
    vectorized = any((np.size(i) > 1 for i in [count1, nobs1, count2, nobs2]))
    if vectorized:
        raise ValueError('function is not vectorized')
    nobs_col = np.array([count1 + count2, nobs1 - count1 + nobs2 - count2])
    nobs_row = np.array([nobs1, nobs2])
    nobs = nobs1 + nobs2
    prob_indep = nobs_col * nobs_row[:, None] / nobs ** 2
    corr = shrink_factor * prob_indep
    if return_corr:
        return (corr[0, 0], corr[0].sum(), corr[1, 0], corr[1].sum())
    else:
        return ((count1 + corr[0, 0], nobs1 + corr[0].sum(), count2 + corr[1, 0], nobs2 + corr[1].sum()), prob_indep)

def score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=None, compare='diff', alternative='two-sided', correction=True, return_results=True):
    if False:
        while True:
            i = 10
    "\n    Score test for two independent proportions\n\n    This uses the constrained estimate of the proportions to compute\n    the variance under the Null hypothesis.\n\n    Parameters\n    ----------\n    count1, nobs1 :\n        count and sample size for first sample\n    count2, nobs2 :\n        count and sample size for the second sample\n    value : float\n        diff, ratio or odds-ratio under the null hypothesis. If value is None,\n        then equality of proportions under the Null is assumed,\n        i.e. value=0 for 'diff' or value=1 for either rate or odds-ratio.\n    compare : string in ['diff', 'ratio' 'odds-ratio']\n        If compare is diff, then the confidence interval is for diff = p1 - p2.\n        If compare is ratio, then the confidence interval is for the risk ratio\n        defined by ratio = p1 / p2.\n        If compare is odds-ratio, then the confidence interval is for the\n        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2)\n    return_results : bool\n        If true, then a results instance with extra information is returned,\n        otherwise a tuple with statistic and pvalue is returned.\n\n    Returns\n    -------\n    results : results instance or tuple\n        If return_results is True, then a results instance with the\n        information in attributes is returned.\n        If return_results is False, then only ``statistic`` and ``pvalue``\n        are returned.\n\n        statistic : float\n            test statistic asymptotically normal distributed N(0, 1)\n        pvalue : float\n            p-value based on normal distribution\n        other attributes :\n            additional information about the hypothesis test\n\n    Notes\n    -----\n    Status: experimental, the type or extra information in the return might\n    change.\n\n    "
    value_default = 0 if compare == 'diff' else 1
    if value is None:
        value = value_default
    nobs = nobs1 + nobs2
    count = count1 + count2
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    if value == value_default:
        prop0 = prop1 = count / nobs
    (count0, nobs0) = (count2, nobs2)
    p0 = p2
    if compare == 'diff':
        diff = value
        if diff != 0:
            tmp3 = nobs
            tmp2 = (nobs1 + 2 * nobs0) * diff - nobs - count
            tmp1 = (count0 * diff - nobs - 2 * count0) * diff + count
            tmp0 = count0 * diff * (1 - diff)
            q = (tmp2 / (3 * tmp3)) ** 3 - tmp1 * tmp2 / (6 * tmp3 ** 2) + tmp0 / (2 * tmp3)
            p = np.sign(q) * np.sqrt((tmp2 / (3 * tmp3)) ** 2 - tmp1 / (3 * tmp3))
            a = (np.pi + np.arccos(q / p ** 3)) / 3
            prop0 = 2 * p * np.cos(a) - tmp2 / (3 * tmp3)
            prop1 = prop0 + diff
        var = prop1 * (1 - prop1) / nobs1 + prop0 * (1 - prop0) / nobs0
        if correction:
            var *= nobs / (nobs - 1)
        diff_stat = p1 - p0 - diff
    elif compare == 'ratio':
        ratio = value
        if ratio != 1:
            a = nobs * ratio
            b = -(nobs1 * ratio + count1 + nobs2 + count0 * ratio)
            c = count
            prop0 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            prop1 = prop0 * ratio
        var = prop1 * (1 - prop1) / nobs1 + ratio ** 2 * prop0 * (1 - prop0) / nobs0
        if correction:
            var *= nobs / (nobs - 1)
        diff_stat = p1 - ratio * p0
    elif compare in ['or', 'odds-ratio']:
        oratio = value
        if oratio != 1:
            a = nobs0 * (oratio - 1)
            b = nobs1 * oratio + nobs0 - count * (oratio - 1)
            c = -count
            prop0 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            prop1 = prop0 * oratio / (1 + prop0 * (oratio - 1))
        eps = 1e-10
        prop0 = np.clip(prop0, eps, 1 - eps)
        prop1 = np.clip(prop1, eps, 1 - eps)
        var = 1 / (prop1 * (1 - prop1) * nobs1) + 1 / (prop0 * (1 - prop0) * nobs0)
        if correction:
            var *= nobs / (nobs - 1)
        diff_stat = (p1 - prop1) / (prop1 * (1 - prop1)) - (p0 - prop0) / (prop0 * (1 - prop0))
    (statistic, pvalue) = _zstat_generic2(diff_stat, np.sqrt(var), alternative=alternative)
    if return_results:
        res = HolderTuple(statistic=statistic, pvalue=pvalue, compare=compare, method='score', variance=var, alternative=alternative, prop1_null=prop1, prop2_null=prop0)
        return res
    else:
        return (statistic, pvalue)

def test_proportions_2indep(count1, nobs1, count2, nobs2, value=None, method=None, compare='diff', alternative='two-sided', correction=True, return_results=True):
    if False:
        return 10
    "\n    Hypothesis test for comparing two independent proportions\n\n    This assumes that we have two independent binomial samples.\n\n    The Null and alternative hypothesis are\n\n    for compare = 'diff'\n\n    - H0: prop1 - prop2 - value = 0\n    - H1: prop1 - prop2 - value != 0  if alternative = 'two-sided'\n    - H1: prop1 - prop2 - value > 0   if alternative = 'larger'\n    - H1: prop1 - prop2 - value < 0   if alternative = 'smaller'\n\n    for compare = 'ratio'\n\n    - H0: prop1 / prop2 - value = 0\n    - H1: prop1 / prop2 - value != 0  if alternative = 'two-sided'\n    - H1: prop1 / prop2 - value > 0   if alternative = 'larger'\n    - H1: prop1 / prop2 - value < 0   if alternative = 'smaller'\n\n    for compare = 'odds-ratio'\n\n    - H0: or - value = 0\n    - H1: or - value != 0  if alternative = 'two-sided'\n    - H1: or - value > 0   if alternative = 'larger'\n    - H1: or - value < 0   if alternative = 'smaller'\n\n    where odds-ratio or = prop1 / (1 - prop1) / (prop2 / (1 - prop2))\n\n    Parameters\n    ----------\n    count1 : int\n        Count for first sample.\n    nobs1 : int\n        Sample size for first sample.\n    count2 : int\n        Count for the second sample.\n    nobs2 : int\n        Sample size for the second sample.\n    value : float\n        Value of the difference, risk ratio or odds ratio of 2 independent\n        proportions under the null hypothesis.\n        Default is equal proportions, 0 for diff and 1 for risk-ratio and for\n        odds-ratio.\n    method : string\n        Method for computing the hypothesis test. If method is None, then a\n        default method is used. The default might change as more methods are\n        added.\n\n        diff:\n\n        - 'wald',\n        - 'agresti-caffo'\n        - 'score' if correction is True, then this uses the degrees of freedom\n           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985\n\n        ratio:\n\n        - 'log': wald test using log transformation\n        - 'log-adjusted': wald test using log transformation,\n           adds 0.5 to counts\n        - 'score': if correction is True, then this uses the degrees of freedom\n           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985\n\n        odds-ratio:\n\n        - 'logit': wald test using logit transformation\n        - 'logit-adjusted': wald test using logit transformation,\n           adds 0.5 to counts\n        - 'logit-smoothed': wald test using logit transformation, biases\n           cell counts towards independence by adding two observations in\n           total.\n        - 'score' if correction is True, then this uses the degrees of freedom\n           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985\n\n    compare : {'diff', 'ratio' 'odds-ratio'}\n        If compare is `diff`, then the hypothesis test is for the risk\n        difference diff = p1 - p2.\n        If compare is `ratio`, then the hypothesis test is for the\n        risk ratio defined by ratio = p1 / p2.\n        If compare is `odds-ratio`, then the hypothesis test is for the\n        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2)\n    alternative : {'two-sided', 'smaller', 'larger'}\n        alternative hypothesis, which can be two-sided or either one of the\n        one-sided tests.\n    correction : bool\n        If correction is True (default), then the Miettinen and Nurminen\n        small sample correction to the variance nobs / (nobs - 1) is used.\n        Applies only if method='score'.\n    return_results : bool\n        If true, then a results instance with extra information is returned,\n        otherwise a tuple with statistic and pvalue is returned.\n\n    Returns\n    -------\n    results : results instance or tuple\n        If return_results is True, then a results instance with the\n        information in attributes is returned.\n        If return_results is False, then only ``statistic`` and ``pvalue``\n        are returned.\n\n        statistic : float\n            test statistic asymptotically normal distributed N(0, 1)\n        pvalue : float\n            p-value based on normal distribution\n        other attributes :\n            additional information about the hypothesis test\n\n    See Also\n    --------\n    tost_proportions_2indep\n    confint_proportions_2indep\n\n    Notes\n    -----\n    Status: experimental, API and defaults might still change.\n        More ``methods`` will be added.\n\n    The current default methods are\n\n    - 'diff': 'agresti-caffo',\n    - 'ratio': 'log-adjusted',\n    - 'odds-ratio': 'logit-adjusted'\n\n    "
    method_default = {'diff': 'agresti-caffo', 'ratio': 'log-adjusted', 'odds-ratio': 'logit-adjusted'}
    if compare.lower() == 'or':
        compare = 'odds-ratio'
    if method is None:
        method = method_default[compare]
    method = method.lower()
    if method.startswith('agr'):
        method = 'agresti-caffo'
    if value is None:
        value = 0 if compare == 'diff' else 1
    (count1, nobs1, count2, nobs2) = map(np.asarray, [count1, nobs1, count2, nobs2])
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    diff = p1 - p2
    ratio = p1 / p2
    odds_ratio = p1 / (1 - p1) / p2 * (1 - p2)
    res = None
    if compare == 'diff':
        if method in ['wald', 'agresti-caffo']:
            addone = 1 if method == 'agresti-caffo' else 0
            (count1_, nobs1_) = (count1 + addone, nobs1 + 2 * addone)
            (count2_, nobs2_) = (count2 + addone, nobs2 + 2 * addone)
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            diff_stat = p1_ - p2_ - value
            var = p1_ * (1 - p1_) / nobs1_ + p2_ * (1 - p2_) / nobs2_
            statistic = diff_stat / np.sqrt(var)
            distr = 'normal'
        elif method.startswith('newcomb'):
            msg = 'newcomb not available for hypothesis test'
            raise NotImplementedError(msg)
        elif method == 'score':
            res = score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=value, compare=compare, alternative=alternative, correction=correction, return_results=return_results)
            if return_results is False:
                (statistic, pvalue) = res[:2]
            distr = 'normal'
            diff_stat = None
        else:
            raise ValueError('method not recognized')
    elif compare == 'ratio':
        if method in ['log', 'log-adjusted']:
            addhalf = 0.5 if method == 'log-adjusted' else 0
            (count1_, nobs1_) = (count1 + addhalf, nobs1 + addhalf)
            (count2_, nobs2_) = (count2 + addhalf, nobs2 + addhalf)
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            ratio_ = p1_ / p2_
            var = 1 / count1_ - 1 / nobs1_ + 1 / count2_ - 1 / nobs2_
            diff_stat = np.log(ratio_) - np.log(value)
            statistic = diff_stat / np.sqrt(var)
            distr = 'normal'
        elif method == 'score':
            res = score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=value, compare=compare, alternative=alternative, correction=correction, return_results=return_results)
            if return_results is False:
                (statistic, pvalue) = res[:2]
            distr = 'normal'
            diff_stat = None
        else:
            raise ValueError('method not recognized')
    elif compare == 'odds-ratio':
        if method in ['logit', 'logit-adjusted', 'logit-smoothed']:
            if method in ['logit-smoothed']:
                adjusted = _shrink_prob(count1, nobs1, count2, nobs2, shrink_factor=2, return_corr=False)[0]
                (count1_, nobs1_, count2_, nobs2_) = adjusted
            else:
                addhalf = 0.5 if method == 'logit-adjusted' else 0
                (count1_, nobs1_) = (count1 + addhalf, nobs1 + 2 * addhalf)
                (count2_, nobs2_) = (count2 + addhalf, nobs2 + 2 * addhalf)
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            odds_ratio_ = p1_ / (1 - p1_) / p2_ * (1 - p2_)
            var = 1 / count1_ + 1 / (nobs1_ - count1_) + 1 / count2_ + 1 / (nobs2_ - count2_)
            diff_stat = np.log(odds_ratio_) - np.log(value)
            statistic = diff_stat / np.sqrt(var)
            distr = 'normal'
        elif method == 'score':
            res = score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=value, compare=compare, alternative=alternative, correction=correction, return_results=return_results)
            if return_results is False:
                (statistic, pvalue) = res[:2]
            distr = 'normal'
            diff_stat = None
        else:
            raise ValueError('method "%s" not recognized' % method)
    else:
        raise ValueError('compare "%s" not recognized' % compare)
    if distr == 'normal' and diff_stat is not None:
        (statistic, pvalue) = _zstat_generic2(diff_stat, np.sqrt(var), alternative=alternative)
    if return_results:
        if res is None:
            res = HolderTuple(statistic=statistic, pvalue=pvalue, compare=compare, method=method, diff=diff, ratio=ratio, odds_ratio=odds_ratio, variance=var, alternative=alternative, value=value)
        else:
            res.diff = diff
            res.ratio = ratio
            res.odds_ratio = odds_ratio
            res.value = value
        return res
    else:
        return (statistic, pvalue)

def tost_proportions_2indep(count1, nobs1, count2, nobs2, low, upp, method=None, compare='diff', correction=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Equivalence test based on two one-sided `test_proportions_2indep`\n\n    This assumes that we have two independent binomial samples.\n\n    The Null and alternative hypothesis for equivalence testing are\n\n    for compare = 'diff'\n\n    - H0: prop1 - prop2 <= low or upp <= prop1 - prop2\n    - H1: low < prop1 - prop2 < upp\n\n    for compare = 'ratio'\n\n    - H0: prop1 / prop2 <= low or upp <= prop1 / prop2\n    - H1: low < prop1 / prop2 < upp\n\n\n    for compare = 'odds-ratio'\n\n    - H0: or <= low or upp <= or\n    - H1: low < or < upp\n\n    where odds-ratio or = prop1 / (1 - prop1) / (prop2 / (1 - prop2))\n\n    Parameters\n    ----------\n    count1, nobs1 :\n        count and sample size for first sample\n    count2, nobs2 :\n        count and sample size for the second sample\n    low, upp :\n        equivalence margin for diff, risk ratio or odds ratio\n    method : string\n        method for computing the hypothesis test. If method is None, then a\n        default method is used. The default might change as more methods are\n        added.\n\n        diff:\n         - 'wald',\n         - 'agresti-caffo'\n         - 'score' if correction is True, then this uses the degrees of freedom\n           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985.\n\n        ratio:\n         - 'log': wald test using log transformation\n         - 'log-adjusted': wald test using log transformation,\n            adds 0.5 to counts\n         - 'score' if correction is True, then this uses the degrees of freedom\n           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985.\n\n        odds-ratio:\n         - 'logit': wald test using logit transformation\n         - 'logit-adjusted': : wald test using logit transformation,\n            adds 0.5 to counts\n         - 'logit-smoothed': : wald test using logit transformation, biases\n            cell counts towards independence by adding two observations in\n            total.\n         - 'score' if correction is True, then this uses the degrees of freedom\n            correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985\n\n    compare : string in ['diff', 'ratio' 'odds-ratio']\n        If compare is `diff`, then the hypothesis test is for\n        diff = p1 - p2.\n        If compare is `ratio`, then the hypothesis test is for the\n        risk ratio defined by ratio = p1 / p2.\n        If compare is `odds-ratio`, then the hypothesis test is for the\n        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).\n    correction : bool\n        If correction is True (default), then the Miettinen and Nurminen\n        small sample correction to the variance nobs / (nobs - 1) is used.\n        Applies only if method='score'.\n\n    Returns\n    -------\n    pvalue : float\n        p-value is the max of the pvalues of the two one-sided tests\n    t1 : test results\n        results instance for one-sided hypothesis at the lower margin\n    t1 : test results\n        results instance for one-sided hypothesis at the upper margin\n\n    See Also\n    --------\n    test_proportions_2indep\n    confint_proportions_2indep\n\n    Notes\n    -----\n    Status: experimental, API and defaults might still change.\n\n    The TOST equivalence test delegates to `test_proportions_2indep` and has\n    the same method and comparison options.\n\n    "
    tt1 = test_proportions_2indep(count1, nobs1, count2, nobs2, value=low, method=method, compare=compare, alternative='larger', correction=correction, return_results=True)
    tt2 = test_proportions_2indep(count1, nobs1, count2, nobs2, value=upp, method=method, compare=compare, alternative='smaller', correction=correction, return_results=True)
    idx_max = np.asarray(tt1.pvalue < tt2.pvalue, int)
    statistic = np.choose(idx_max, [tt1.statistic, tt2.statistic])
    pvalue = np.choose(idx_max, [tt1.pvalue, tt2.pvalue])
    res = HolderTuple(statistic=statistic, pvalue=pvalue, compare=compare, method=method, results_larger=tt1, results_smaller=tt2, title='Equivalence test for 2 independent proportions')
    return res

def _std_2prop_power(diff, p2, ratio=1, alpha=0.05, value=0):
    if False:
        while True:
            i = 10
    '\n    Compute standard error under null and alternative for 2 proportions\n\n    helper function for power and sample size computation\n\n    '
    if value != 0:
        msg = 'non-zero diff under null, value, is not yet implemented'
        raise NotImplementedError(msg)
    nobs_ratio = ratio
    p1 = p2 + diff
    p_pooled = (p1 + p2 * ratio) / (1 + ratio)
    (p1_vnull, p2_vnull) = (p_pooled, p_pooled)
    p2_alt = p2
    p1_alt = p2_alt + diff
    std_null = _std_diff_prop(p1_vnull, p2_vnull, ratio=nobs_ratio)
    std_alt = _std_diff_prop(p1_alt, p2_alt, ratio=nobs_ratio)
    return (p_pooled, std_null, std_alt)

def power_proportions_2indep(diff, prop2, nobs1, ratio=1, alpha=0.05, value=0, alternative='two-sided', return_results=True):
    if False:
        while True:
            i = 10
    "\n    Power for ztest that two independent proportions are equal\n\n    This assumes that the variance is based on the pooled proportion\n    under the null and the non-pooled variance under the alternative\n\n    Parameters\n    ----------\n    diff : float\n        difference between proportion 1 and 2 under the alternative\n    prop2 : float\n        proportion for the reference case, prop2, proportions for the\n        first case will be computed using p2 and diff\n        p1 = p2 + diff\n    nobs1 : float or int\n        number of observations in sample 1\n    ratio : float\n        sample size ratio, nobs2 = ratio * nobs1\n    alpha : float in interval (0,1)\n        Significance level, e.g. 0.05, is the probability of a type I\n        error, that is wrong rejections if the Null Hypothesis is true.\n    value : float\n        currently only `value=0`, i.e. equality testing, is supported\n    alternative : string, 'two-sided' (default), 'larger', 'smaller'\n        Alternative hypothesis whether the power is calculated for a\n        two-sided (default) or one sided test. The one-sided test can be\n        either 'larger', 'smaller'.\n    return_results : bool\n        If true, then a results instance with extra information is returned,\n        otherwise only the computed power is returned.\n\n    Returns\n    -------\n    results : results instance or float\n        If return_results is True, then a results instance with the\n        information in attributes is returned.\n        If return_results is False, then only the power is returned.\n\n        power : float\n            Power of the test, e.g. 0.8, is one minus the probability of a\n            type II error. Power is the probability that the test correctly\n            rejects the Null Hypothesis if the Alternative Hypothesis is true.\n\n        Other attributes in results instance include :\n\n        p_pooled\n            pooled proportion, used for std_null\n        std_null\n            standard error of difference under the null hypothesis (without\n            sqrt(nobs1))\n        std_alt\n            standard error of difference under the alternative hypothesis\n            (without sqrt(nobs1))\n    "
    from statsmodels.stats.power import normal_power_het
    (p_pooled, std_null, std_alt) = _std_2prop_power(diff, prop2, ratio=ratio, alpha=alpha, value=value)
    pow_ = normal_power_het(diff, nobs1, alpha, std_null=std_null, std_alternative=std_alt, alternative=alternative)
    if return_results:
        res = Holder(power=pow_, p_pooled=p_pooled, std_null=std_null, std_alt=std_alt, nobs1=nobs1, nobs2=ratio * nobs1, nobs_ratio=ratio, alpha=alpha)
        return res
    else:
        return pow_

def samplesize_proportions_2indep_onetail(diff, prop2, power, ratio=1, alpha=0.05, value=0, alternative='two-sided'):
    if False:
        return 10
    "\n    Required sample size assuming normal distribution based on one tail\n\n    This uses an explicit computation for the sample size that is required\n    to achieve a given power corresponding to the appropriate tails of the\n    normal distribution. This ignores the far tail in a two-sided test\n    which is negligible in the common case when alternative and null are\n    far apart.\n\n    Parameters\n    ----------\n    diff : float\n        Difference between proportion 1 and 2 under the alternative\n    prop2 : float\n        proportion for the reference case, prop2, proportions for the\n        first case will be computing using p2 and diff\n        p1 = p2 + diff\n    power : float\n        Power for which sample size is computed.\n    ratio : float\n        Sample size ratio, nobs2 = ratio * nobs1\n    alpha : float in interval (0,1)\n        Significance level, e.g. 0.05, is the probability of a type I\n        error, that is wrong rejections if the Null Hypothesis is true.\n    value : float\n        Currently only `value=0`, i.e. equality testing, is supported\n    alternative : string, 'two-sided' (default), 'larger', 'smaller'\n        Alternative hypothesis whether the power is calculated for a\n        two-sided (default) or one sided test. In the case of a one-sided\n        alternative, it is assumed that the test is in the appropriate tail.\n\n    Returns\n    -------\n    nobs1 : float\n        Number of observations in sample 1.\n    "
    from statsmodels.stats.power import normal_sample_size_one_tail
    if alternative in ['two-sided', '2s']:
        alpha = alpha / 2
    (_, std_null, std_alt) = _std_2prop_power(diff, prop2, ratio=ratio, alpha=alpha, value=value)
    nobs = normal_sample_size_one_tail(diff, power, alpha, std_null=std_null, std_alternative=std_alt)
    return nobs

def _score_confint_inversion(count1, nobs1, count2, nobs2, compare='diff', alpha=0.05, correction=True):
    if False:
        return 10
    "\n    Compute score confidence interval by inverting score test\n\n    Parameters\n    ----------\n    count1, nobs1 :\n        Count and sample size for first sample.\n    count2, nobs2 :\n        Count and sample size for the second sample.\n    compare : string in ['diff', 'ratio' 'odds-ratio']\n        If compare is `diff`, then the confidence interval is for\n        diff = p1 - p2.\n        If compare is `ratio`, then the confidence interval is for the\n        risk ratio defined by ratio = p1 / p2.\n        If compare is `odds-ratio`, then the confidence interval is for the\n        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).\n    alpha : float in interval (0,1)\n        Significance level, e.g. 0.05, is the probability of a type I\n        error, that is wrong rejections if the Null Hypothesis is true.\n    correction : bool\n        If correction is True (default), then the Miettinen and Nurminen\n        small sample correction to the variance nobs / (nobs - 1) is used.\n        Applies only if method='score'.\n\n    Returns\n    -------\n    low : float\n        Lower confidence bound.\n    upp : float\n        Upper confidence bound.\n    "

    def func(v):
        if False:
            while True:
                i = 10
        r = test_proportions_2indep(count1, nobs1, count2, nobs2, value=v, compare=compare, method='score', correction=correction, alternative='two-sided')
        return r.pvalue - alpha
    rt0 = test_proportions_2indep(count1, nobs1, count2, nobs2, value=0, compare=compare, method='score', correction=correction, alternative='two-sided')
    use_method = {'diff': 'wald', 'ratio': 'log', 'odds-ratio': 'logit'}
    rci0 = confint_proportions_2indep(count1, nobs1, count2, nobs2, method=use_method[compare], compare=compare, alpha=alpha)
    ub = rci0[1] + np.abs(rci0[1]) * 0.5
    lb = rci0[0] - np.abs(rci0[0]) * 0.25
    if compare == 'diff':
        param = rt0.diff
        ub = min(ub, 0.99999)
    elif compare == 'ratio':
        param = rt0.ratio
        ub *= 2
    if compare == 'odds-ratio':
        param = rt0.odds_ratio
    upp = optimize.brentq(func, param, ub)
    low = optimize.brentq(func, lb, param)
    return (low, upp)

def _confint_riskratio_koopman(count1, nobs1, count2, nobs2, alpha=0.05, correction=True):
    if False:
        print('Hello World!')
    '\n    Score confidence interval for ratio or proportions, Koopman/Nam\n\n    signature not consistent with other functions\n\n    When correction is True, then the small sample correction nobs / (nobs - 1)\n    by Miettinen/Nurminen is used.\n    '
    (x0, x1, n0, n1) = (count2, count1, nobs2, nobs1)
    x = x0 + x1
    n = n0 + n1
    z = stats.norm.isf(alpha / 2) ** 2
    if correction:
        z *= n / (n - 1)
    a1 = n0 * (n0 * n * x1 + n1 * (n0 + x1) * z)
    a2 = -n0 * (n0 * n1 * x + 2 * n * x0 * x1 + n1 * (n0 + x0 + 2 * x1) * z)
    a3 = 2 * n0 * n1 * x0 * x + n * x0 * x0 * x1 + n0 * n1 * x * z
    a4 = -n1 * x0 * x0 * x
    p_roots_ = np.sort(np.roots([a1, a2, a3, a4]))
    p_roots = p_roots_[:2][::-1]
    ci = (1 - (n1 - x1) * (1 - p_roots) / (x0 + n1 - n * p_roots)) / p_roots
    res = Holder()
    res.confint = ci
    res._p_roots = p_roots_
    return res

def _confint_riskratio_paired_nam(table, alpha=0.05):
    if False:
        for i in range(10):
            print('nop')
    '\n    Confidence interval for marginal risk ratio for matched pairs\n\n    need full table\n\n             success fail  marginal\n    success    x11    x10  x1.\n    fail       x01    x00  x0.\n    marginal   x.1    x.0   n\n\n    The confidence interval is for the ratio p1 / p0 where\n    p1 = x1. / n and\n    p0 - x.1 / n\n    Todo: rename p1 to pa and p2 to pb, so we have a, b for treatment and\n    0, 1 for success/failure\n\n    current namings follow Nam 2009\n\n    status\n    testing:\n    compared to example in Nam 2009\n    internal polynomial coefficients in calculation correspond at around\n        4 decimals\n    confidence interval agrees only at 2 decimals\n\n    '
    (x11, x10, x01, x00) = np.ravel(table)
    n = np.sum(table)
    (p10, p01) = (x10 / n, x01 / n)
    p1 = (x11 + x10) / n
    p0 = (x11 + x01) / n
    q00 = 1 - x00 / n
    z2 = stats.norm.isf(alpha / 2) ** 2
    g1 = (n * p0 + z2 / 2) * p0
    g2 = -(2 * n * p1 * p0 + z2 * q00)
    g3 = (n * p1 + z2 / 2) * p1
    a0 = g1 ** 2 - (z2 * p0 / 2) ** 2
    a1 = 2 * g1 * g2
    a2 = g2 ** 2 + 2 * g1 * g3 + z2 ** 2 * (p1 * p0 - 2 * p10 * p01) / 2
    a3 = 2 * g2 * g3
    a4 = g3 ** 2 - (z2 * p1 / 2) ** 2
    p_roots = np.sort(np.roots([a0, a1, a2, a3, a4]))
    ci = [p_roots.min(), p_roots.max()]
    res = Holder()
    res.confint = ci
    res.p = (p1, p0)
    res._p_roots = p_roots
    return res