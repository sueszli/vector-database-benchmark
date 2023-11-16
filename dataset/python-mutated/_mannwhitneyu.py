import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory

def _broadcast_concatenate(x, y, axis):
    if False:
        print('Hello World!')
    'Broadcast then concatenate arrays, leaving concatenation axis last'
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    z = np.broadcast(x[..., 0], y[..., 0])
    x = np.broadcast_to(x, z.shape + (x.shape[-1],))
    y = np.broadcast_to(y, z.shape + (y.shape[-1],))
    z = np.concatenate((x, y), axis=-1)
    return (x, y, z)

class _MWU:
    """Distribution of MWU statistic under the null hypothesis"""

    def __init__(self):
        if False:
            return 10
        'Minimal initializer'
        self._fmnks = -np.ones((1, 1, 1))
        self._recursive = None

    def pmf(self, k, m, n):
        if False:
            while True:
                i = 10
        if self._recursive is None and m <= 500 and (n <= 500) or self._recursive:
            return self.pmf_recursive(k, m, n)
        else:
            return self.pmf_iterative(k, m, n)

    def pmf_recursive(self, k, m, n):
        if False:
            while True:
                i = 10
        'Probability mass function, recursive version'
        self._resize_fmnks(m, n, np.max(k))
        for i in np.ravel(k):
            self._f(m, n, i)
        return self._fmnks[m, n, k] / special.binom(m + n, m)

    def pmf_iterative(self, k, m, n):
        if False:
            print('Hello World!')
        'Probability mass function, iterative version'
        fmnks = {}
        for i in np.ravel(k):
            fmnks = _mwu_f_iterative(m, n, i, fmnks)
        return np.array([fmnks[m, n, ki] for ki in k]) / special.binom(m + n, m)

    def cdf(self, k, m, n):
        if False:
            return 10
        'Cumulative distribution function'
        pmfs = self.pmf(np.arange(0, np.max(k) + 1), m, n)
        cdfs = np.cumsum(pmfs)
        return cdfs[k]

    def sf(self, k, m, n):
        if False:
            return 10
        'Survival function'
        k = m * n - k
        return self.cdf(k, m, n)

    def _resize_fmnks(self, m, n, k):
        if False:
            while True:
                i = 10
        'If necessary, expand the array that remembers PMF values'
        shape_old = np.array(self._fmnks.shape)
        shape_new = np.array((m + 1, n + 1, k + 1))
        if np.any(shape_new > shape_old):
            shape = np.maximum(shape_old, shape_new)
            fmnks = -np.ones(shape)
            (m0, n0, k0) = shape_old
            fmnks[:m0, :n0, :k0] = self._fmnks
            self._fmnks = fmnks

    def _f(self, m, n, k):
        if False:
            return 10
        'Recursive implementation of function of [3] Theorem 2.5'
        if k < 0 or m < 0 or n < 0 or (k > m * n):
            return 0
        if self._fmnks[m, n, k] >= 0:
            return self._fmnks[m, n, k]
        if k == 0 and m >= 0 and (n >= 0):
            fmnk = 1
        else:
            fmnk = self._f(m - 1, n, k - n) + self._f(m, n - 1, k)
        self._fmnks[m, n, k] = fmnk
        return fmnk
_mwu_state = _MWU()

def _mwu_f_iterative(m, n, k, fmnks):
    if False:
        print('Hello World!')
    'Iterative implementation of function of [3] Theorem 2.5'

    def _base_case(m, n, k):
        if False:
            i = 10
            return i + 15
        'Base cases from recursive version'
        if fmnks.get((m, n, k), -1) >= 0:
            return fmnks[m, n, k]
        elif k < 0 or m < 0 or n < 0 or (k > m * n):
            return 0
        elif k == 0 and m >= 0 and (n >= 0):
            return 1
        return None
    stack = [(m, n, k)]
    fmnk = None
    while stack:
        (m, n, k) = stack.pop()
        fmnk = _base_case(m, n, k)
        if fmnk is not None:
            fmnks[m, n, k] = fmnk
            continue
        f1 = _base_case(m - 1, n, k - n)
        f2 = _base_case(m, n - 1, k)
        if f1 is not None and f2 is not None:
            fmnk = f1 + f2
            fmnks[m, n, k] = fmnk
            continue
        stack.append((m, n, k))
        if f1 is None:
            stack.append((m - 1, n, k - n))
        if f2 is None:
            stack.append((m, n - 1, k))
    return fmnks

def _tie_term(ranks):
    if False:
        print('Hello World!')
    'Tie correction term'
    (_, t) = np.unique(ranks, return_counts=True, axis=-1)
    return (t ** 3 - t).sum(axis=-1)

def _get_mwu_z(U, n1, n2, ranks, axis=0, continuity=True):
    if False:
        return 10
    'Standardized MWU statistic'
    mu = n1 * n2 / 2
    n = n1 + n2
    tie_term = np.apply_along_axis(_tie_term, -1, ranks)
    s = np.sqrt(n1 * n2 / 12 * (n + 1 - tie_term / (n * (n - 1))))
    numerator = U - mu
    if continuity:
        numerator -= 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s
    return z

def _mwu_input_validation(x, y, use_continuity, alternative, axis, method):
    if False:
        while True:
            i = 10
    ' Input validation and standardization for mannwhitneyu '
    (x, y) = (np.atleast_1d(x), np.atleast_1d(y))
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')
    bools = {True, False}
    if use_continuity not in bools:
        raise ValueError(f'`use_continuity` must be one of {bools}.')
    alternatives = {'two-sided', 'less', 'greater'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')
    methods = {'asymptotic', 'exact', 'auto'}
    method = method.lower()
    if method not in methods:
        raise ValueError(f'`method` must be one of {methods}.')
    return (x, y, use_continuity, alternative, axis_int, method)

def _tie_check(xy):
    if False:
        return 10
    'Find any ties in data'
    (_, t) = np.unique(xy, return_counts=True, axis=-1)
    return np.any(t != 1)

def _mwu_choose_method(n1, n2, xy, method):
    if False:
        print('Hello World!')
    "Choose method 'asymptotic' or 'exact' depending on input size, ties"
    if n1 > 8 and n2 > 8:
        return 'asymptotic'
    if np.apply_along_axis(_tie_check, -1, xy).any():
        return 'asymptotic'
    return 'exact'
MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))

@_axis_nan_policy_factory(MannwhitneyuResult, n_samples=2)
def mannwhitneyu(x, y, use_continuity=True, alternative='two-sided', axis=0, method='auto'):
    if False:
        print('Hello World!')
    'Perform the Mann-Whitney U rank test on two independent samples.\n\n    The Mann-Whitney U test is a nonparametric test of the null hypothesis\n    that the distribution underlying sample `x` is the same as the\n    distribution underlying sample `y`. It is often used as a test of\n    difference in location between distributions.\n\n    Parameters\n    ----------\n    x, y : array-like\n        N-d arrays of samples. The arrays must be broadcastable except along\n        the dimension given by `axis`.\n    use_continuity : bool, optional\n            Whether a continuity correction (1/2) should be applied.\n            Default is True when `method` is ``\'asymptotic\'``; has no effect\n            otherwise.\n    alternative : {\'two-sided\', \'less\', \'greater\'}, optional\n        Defines the alternative hypothesis. Default is \'two-sided\'.\n        Let *F(u)* and *G(u)* be the cumulative distribution functions of the\n        distributions underlying `x` and `y`, respectively. Then the following\n        alternative hypotheses are available:\n\n        * \'two-sided\': the distributions are not equal, i.e. *F(u) ≠ G(u)* for\n          at least one *u*.\n        * \'less\': the distribution underlying `x` is stochastically less\n          than the distribution underlying `y`, i.e. *F(u) > G(u)* for all *u*.\n        * \'greater\': the distribution underlying `x` is stochastically greater\n          than the distribution underlying `y`, i.e. *F(u) < G(u)* for all *u*.\n\n        Note that the mathematical expressions in the alternative hypotheses\n        above describe the CDFs of the underlying distributions. The directions\n        of the inequalities appear inconsistent with the natural language\n        description at first glance, but they are not. For example, suppose\n        *X* and *Y* are random variables that follow distributions with CDFs\n        *F* and *G*, respectively. If *F(u) > G(u)* for all *u*, samples drawn\n        from *X* tend to be less than those drawn from *Y*.\n\n        Under a more restrictive set of assumptions, the alternative hypotheses\n        can be expressed in terms of the locations of the distributions;\n        see [5] section 5.1.\n    axis : int, optional\n        Axis along which to perform the test. Default is 0.\n    method : {\'auto\', \'asymptotic\', \'exact\'}, optional\n        Selects the method used to calculate the *p*-value.\n        Default is \'auto\'. The following options are available.\n\n        * ``\'asymptotic\'``: compares the standardized test statistic\n          against the normal distribution, correcting for ties.\n        * ``\'exact\'``: computes the exact *p*-value by comparing the observed\n          :math:`U` statistic against the exact distribution of the :math:`U`\n          statistic under the null hypothesis. No correction is made for ties.\n        * ``\'auto\'``: chooses ``\'exact\'`` when the size of one of the samples\n          is less than or equal to 8 and there are no ties;\n          chooses ``\'asymptotic\'`` otherwise.\n\n    Returns\n    -------\n    res : MannwhitneyuResult\n        An object containing attributes:\n\n        statistic : float\n            The Mann-Whitney U statistic corresponding with sample `x`. See\n            Notes for the test statistic corresponding with sample `y`.\n        pvalue : float\n            The associated *p*-value for the chosen `alternative`.\n\n    Notes\n    -----\n    If ``U1`` is the statistic corresponding with sample `x`, then the\n    statistic corresponding with sample `y` is\n    ``U2 = x.shape[axis] * y.shape[axis] - U1``.\n\n    `mannwhitneyu` is for independent samples. For related / paired samples,\n    consider `scipy.stats.wilcoxon`.\n\n    `method` ``\'exact\'`` is recommended when there are no ties and when either\n    sample size is less than 8 [1]_. The implementation follows the recurrence\n    relation originally proposed in [1]_ as it is described in [3]_.\n    Note that the exact method is *not* corrected for ties, but\n    `mannwhitneyu` will not raise errors or warnings if there are ties in the\n    data.\n\n    The Mann-Whitney U test is a non-parametric version of the t-test for\n    independent samples. When the means of samples from the populations\n    are normally distributed, consider `scipy.stats.ttest_ind`.\n\n    See Also\n    --------\n    scipy.stats.wilcoxon, scipy.stats.ranksums, scipy.stats.ttest_ind\n\n    References\n    ----------\n    .. [1] H.B. Mann and D.R. Whitney, "On a test of whether one of two random\n           variables is stochastically larger than the other", The Annals of\n           Mathematical Statistics, Vol. 18, pp. 50-60, 1947.\n    .. [2] Mann-Whitney U Test, Wikipedia,\n           http://en.wikipedia.org/wiki/Mann-Whitney_U_test\n    .. [3] A. Di Bucchianico, "Combinatorics, computer algebra, and the\n           Wilcoxon-Mann-Whitney test", Journal of Statistical Planning and\n           Inference, Vol. 79, pp. 349-364, 1999.\n    .. [4] Rosie Shier, "Statistics: 2.3 The Mann-Whitney U Test", Mathematics\n           Learning Support Centre, 2004.\n    .. [5] Michael P. Fay and Michael A. Proschan. "Wilcoxon-Mann-Whitney\n           or t-test? On assumptions for hypothesis tests and multiple \\\n           interpretations of decision rules." Statistics surveys, Vol. 4, pp.\n           1-39, 2010. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2857732/\n\n    Examples\n    --------\n    We follow the example from [4]_: nine randomly sampled young adults were\n    diagnosed with type II diabetes at the ages below.\n\n    >>> males = [19, 22, 16, 29, 24]\n    >>> females = [20, 11, 17, 12]\n\n    We use the Mann-Whitney U test to assess whether there is a statistically\n    significant difference in the diagnosis age of males and females.\n    The null hypothesis is that the distribution of male diagnosis ages is\n    the same as the distribution of female diagnosis ages. We decide\n    that a confidence level of 95% is required to reject the null hypothesis\n    in favor of the alternative that the distributions are different.\n    Since the number of samples is very small and there are no ties in the\n    data, we can compare the observed test statistic against the *exact*\n    distribution of the test statistic under the null hypothesis.\n\n    >>> from scipy.stats import mannwhitneyu\n    >>> U1, p = mannwhitneyu(males, females, method="exact")\n    >>> print(U1)\n    17.0\n\n    `mannwhitneyu` always reports the statistic associated with the first\n    sample, which, in this case, is males. This agrees with :math:`U_M = 17`\n    reported in [4]_. The statistic associated with the second statistic\n    can be calculated:\n\n    >>> nx, ny = len(males), len(females)\n    >>> U2 = nx*ny - U1\n    >>> print(U2)\n    3.0\n\n    This agrees with :math:`U_F = 3` reported in [4]_. The two-sided\n    *p*-value can be calculated from either statistic, and the value produced\n    by `mannwhitneyu` agrees with :math:`p = 0.11` reported in [4]_.\n\n    >>> print(p)\n    0.1111111111111111\n\n    The exact distribution of the test statistic is asymptotically normal, so\n    the example continues by comparing the exact *p*-value against the\n    *p*-value produced using the normal approximation.\n\n    >>> _, pnorm = mannwhitneyu(males, females, method="asymptotic")\n    >>> print(pnorm)\n    0.11134688653314041\n\n    Here `mannwhitneyu`\'s reported *p*-value appears to conflict with the\n    value :math:`p = 0.09` given in [4]_. The reason is that [4]_\n    does not apply the continuity correction performed by `mannwhitneyu`;\n    `mannwhitneyu` reduces the distance between the test statistic and the\n    mean :math:`\\mu = n_x n_y / 2` by 0.5 to correct for the fact that the\n    discrete statistic is being compared against a continuous distribution.\n    Here, the :math:`U` statistic used is less than the mean, so we reduce\n    the distance by adding 0.5 in the numerator.\n\n    >>> import numpy as np\n    >>> from scipy.stats import norm\n    >>> U = min(U1, U2)\n    >>> N = nx + ny\n    >>> z = (U - nx*ny/2 + 0.5) / np.sqrt(nx*ny * (N + 1)/ 12)\n    >>> p = 2 * norm.cdf(z)  # use CDF to get p-value from smaller statistic\n    >>> print(p)\n    0.11134688653314041\n\n    If desired, we can disable the continuity correction to get a result\n    that agrees with that reported in [4]_.\n\n    >>> _, pnorm = mannwhitneyu(males, females, use_continuity=False,\n    ...                         method="asymptotic")\n    >>> print(pnorm)\n    0.0864107329737\n\n    Regardless of whether we perform an exact or asymptotic test, the\n    probability of the test statistic being as extreme or more extreme by\n    chance exceeds 5%, so we do not consider the results statistically\n    significant.\n\n    Suppose that, before seeing the data, we had hypothesized that females\n    would tend to be diagnosed at a younger age than males.\n    In that case, it would be natural to provide the female ages as the\n    first input, and we would have performed a one-sided test using\n    ``alternative = \'less\'``: females are diagnosed at an age that is\n    stochastically less than that of males.\n\n    >>> res = mannwhitneyu(females, males, alternative="less", method="exact")\n    >>> print(res)\n    MannwhitneyuResult(statistic=3.0, pvalue=0.05555555555555555)\n\n    Again, the probability of getting a sufficiently low value of the\n    test statistic by chance under the null hypothesis is greater than 5%,\n    so we do not reject the null hypothesis in favor of our alternative.\n\n    If it is reasonable to assume that the means of samples from the\n    populations are normally distributed, we could have used a t-test to\n    perform the analysis.\n\n    >>> from scipy.stats import ttest_ind\n    >>> res = ttest_ind(females, males, alternative="less")\n    >>> print(res)\n    Ttest_indResult(statistic=-2.239334696520584, pvalue=0.030068441095757924)\n\n    Under this assumption, the *p*-value would be low enough to reject the\n    null hypothesis in favor of the alternative.\n\n    '
    (x, y, use_continuity, alternative, axis_int, method) = _mwu_input_validation(x, y, use_continuity, alternative, axis, method)
    (x, y, xy) = _broadcast_concatenate(x, y, axis)
    (n1, n2) = (x.shape[-1], y.shape[-1])
    if method == 'auto':
        method = _mwu_choose_method(n1, n2, xy, method)
    ranks = stats.rankdata(xy, axis=-1)
    R1 = ranks[..., :n1].sum(axis=-1)
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    if alternative == 'greater':
        (U, f) = (U1, 1)
    elif alternative == 'less':
        (U, f) = (U2, 1)
    else:
        (U, f) = (np.maximum(U1, U2), 2)
    if method == 'exact':
        p = _mwu_state.sf(U.astype(int), n1, n2)
    elif method == 'asymptotic':
        z = _get_mwu_z(U, n1, n2, ranks, continuity=use_continuity)
        p = stats.norm.sf(z)
    p *= f
    p = np.clip(p, 0, 1)
    return MannwhitneyuResult(U1, p)