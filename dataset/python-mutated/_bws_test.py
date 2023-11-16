import numpy as np
from functools import partial
from scipy import stats

def _bws_input_validation(x, y, alternative, method):
    if False:
        return 10
    ' Input validation and standardization for bws test'
    (x, y) = np.atleast_1d(x, y)
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError('`x` and `y` must be exactly one-dimensional.')
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')
    z = stats.rankdata(np.concatenate((x, y)))
    (x, y) = (z[:len(x)], z[len(x):])
    alternatives = {'two-sided', 'less', 'greater'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')
    method = stats.PermutationMethod() if method is None else method
    if not isinstance(method, stats.PermutationMethod):
        raise ValueError('`method` must be an instance of `scipy.stats.PermutationMethod`')
    return (x, y, alternative, method)

def _bws_statistic(x, y, alternative, axis):
    if False:
        print('Hello World!')
    'Compute the BWS test statistic for two independent samples'
    (Ri, Hj) = (np.sort(x, axis=axis), np.sort(y, axis=axis))
    (n, m) = (Ri.shape[axis], Hj.shape[axis])
    (i, j) = (np.arange(1, n + 1), np.arange(1, m + 1))
    Bx_num = Ri - (m + n) / n * i
    By_num = Hj - (m + n) / m * j
    if alternative == 'two-sided':
        Bx_num *= Bx_num
        By_num *= By_num
    else:
        Bx_num *= np.abs(Bx_num)
        By_num *= np.abs(By_num)
    Bx_den = i / (n + 1) * (1 - i / (n + 1)) * m * (m + n) / n
    By_den = j / (m + 1) * (1 - j / (m + 1)) * n * (m + n) / m
    Bx = 1 / n * np.sum(Bx_num / Bx_den, axis=axis)
    By = 1 / m * np.sum(By_num / By_den, axis=axis)
    B = (Bx + By) / 2 if alternative == 'two-sided' else (Bx - By) / 2
    return B

def bws_test(x, y, *, alternative='two-sided', method=None):
    if False:
        while True:
            i = 10
    "Perform the Baumgartner-Weiss-Schindler test on two independent samples.\n\n    The Baumgartner-Weiss-Schindler (BWS) test is a nonparametric test of \n    the null hypothesis that the distribution underlying sample `x` \n    is the same as the distribution underlying sample `y`. Unlike \n    the Kolmogorov-Smirnov, Wilcoxon, and Cramer-Von Mises tests, \n    the BWS test weights the integral by the variance of the difference\n    in cumulative distribution functions (CDFs), emphasizing the tails of the\n    distributions, which increases the power of the test in many applications.\n\n    Parameters\n    ----------\n    x, y : array-like\n        1-d arrays of samples.\n    alternative : {'two-sided', 'less', 'greater'}, optional\n        Defines the alternative hypothesis. Default is 'two-sided'.\n        Let *F(u)* and *G(u)* be the cumulative distribution functions of the\n        distributions underlying `x` and `y`, respectively. Then the following\n        alternative hypotheses are available:\n\n        * 'two-sided': the distributions are not equal, i.e. *F(u) ≠ G(u)* for\n          at least one *u*.\n        * 'less': the distribution underlying `x` is stochastically less than\n          the distribution underlying `y`, i.e. *F(u) >= G(u)* for all *u*.\n        * 'greater': the distribution underlying `x` is stochastically greater\n          than the distribution underlying `y`, i.e. *F(u) <= G(u)* for all\n          *u*.\n\n        Under a more restrictive set of assumptions, the alternative hypotheses\n        can be expressed in terms of the locations of the distributions;\n        see [2] section 5.1.\n    method : PermutationMethod, optional\n        Configures the method used to compute the p-value. The default is\n        the default `PermutationMethod` object.\n\n    Returns\n    -------\n    res : PermutationTestResult\n    An object with attributes:\n\n    statistic : float\n        The observed test statistic of the data.\n    pvalue : float\n        The p-value for the given alternative.\n    null_distribution : ndarray\n        The values of the test statistic generated under the null hypothesis.\n\n    See also\n    --------\n    scipy.stats.wilcoxon, scipy.stats.mannwhitneyu, scipy.stats.ttest_ind\n\n    Notes\n    -----\n    When ``alternative=='two-sided'``, the statistic is defined by the\n    equations given in [1]_ Section 2. This statistic is not appropriate for\n    one-sided alternatives; in that case, the statistic is the *negative* of\n    that given by the equations in [1]_ Section 2. Consequently, when the\n    distribution of the first sample is stochastically greater than that of the\n    second sample, the statistic will tend to be positive.\n\n    References\n    ----------\n    .. [1] Neuhäuser, M. (2005). Exact Tests Based on the\n           Baumgartner-Weiss-Schindler Statistic: A Survey. Statistical Papers,\n           46(1), 1-29.\n    .. [2] Fay, M. P., & Proschan, M. A. (2010). Wilcoxon-Mann-Whitney or t-test?\n           On assumptions for hypothesis tests and multiple interpretations of \n           decision rules. Statistics surveys, 4, 1.\n\n    Examples\n    --------\n    We follow the example of table 3 in [1]_: Fourteen children were divided\n    randomly into two groups. Their ranks at performing a specific tests are\n    as follows.\n\n    >>> import numpy as np\n    >>> x = [1, 2, 3, 4, 6, 7, 8]\n    >>> y = [5, 9, 10, 11, 12, 13, 14]\n\n    We use the BWS test to assess whether there is a statistically significant\n    difference between the two groups.\n    The null hypothesis is that there is no difference in the distributions of\n    performance between the two groups. We decide that a significance level of\n    1% is required to reject the null hypothesis in favor of the alternative\n    that the distributions are different.\n    Since the number of samples is very small, we can compare the observed test\n    statistic against the *exact* distribution of the test statistic under the\n    null hypothesis.\n\n    >>> from scipy.stats import bws_test\n    >>> res = bws_test(x, y)\n    >>> print(res.statistic)\n    5.132167152575315\n\n    This agrees with :math:`B = 5.132` reported in [1]_. The *p*-value produced\n    by `bws_test` also agrees with :math:`p = 0.0029` reported in [1]_.\n\n    >>> print(res.pvalue)\n    0.002913752913752914\n\n    Because the p-value is below our threshold of 1%, we take this as evidence\n    against the null hypothesis in favor of the alternative that there is a\n    difference in performance between the two groups.\n    "
    (x, y, alternative, method) = _bws_input_validation(x, y, alternative, method)
    bws_statistic = partial(_bws_statistic, alternative=alternative)
    permutation_alternative = 'less' if alternative == 'less' else 'greater'
    res = stats.permutation_test((x, y), bws_statistic, alternative=permutation_alternative, **method._asdict())
    return res