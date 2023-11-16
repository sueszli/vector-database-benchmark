from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass

@dataclass
class PageTrendTestResult:
    statistic: float
    pvalue: float
    method: str

def page_trend_test(data, ranked=False, predicted_ranks=None, method='auto'):
    if False:
        return 10
    '\n    Perform Page\'s Test, a measure of trend in observations between treatments.\n\n    Page\'s Test (also known as Page\'s :math:`L` test) is useful when:\n\n    * there are :math:`n \\geq 3` treatments,\n    * :math:`m \\geq 2` subjects are observed for each treatment, and\n    * the observations are hypothesized to have a particular order.\n\n    Specifically, the test considers the null hypothesis that\n\n    .. math::\n\n        m_1 = m_2 = m_3 \\cdots = m_n,\n\n    where :math:`m_j` is the mean of the observed quantity under treatment\n    :math:`j`, against the alternative hypothesis that\n\n    .. math::\n\n        m_1 \\leq m_2 \\leq m_3 \\leq \\cdots \\leq m_n,\n\n    where at least one inequality is strict.\n\n    As noted by [4]_, Page\'s :math:`L` test has greater statistical power than\n    the Friedman test against the alternative that there is a difference in\n    trend, as Friedman\'s test only considers a difference in the means of the\n    observations without considering their order. Whereas Spearman :math:`\\rho`\n    considers the correlation between the ranked observations of two variables\n    (e.g. the airspeed velocity of a swallow vs. the weight of the coconut it\n    carries), Page\'s :math:`L` is concerned with a trend in an observation\n    (e.g. the airspeed velocity of a swallow) across several distinct\n    treatments (e.g. carrying each of five coconuts of different weight) even\n    as the observation is repeated with multiple subjects (e.g. one European\n    swallow and one African swallow).\n\n    Parameters\n    ----------\n    data : array-like\n        A :math:`m \\times n` array; the element in row :math:`i` and\n        column :math:`j` is the observation corresponding with subject\n        :math:`i` and treatment :math:`j`. By default, the columns are\n        assumed to be arranged in order of increasing predicted mean.\n\n    ranked : boolean, optional\n        By default, `data` is assumed to be observations rather than ranks;\n        it will be ranked with `scipy.stats.rankdata` along ``axis=1``. If\n        `data` is provided in the form of ranks, pass argument ``True``.\n\n    predicted_ranks : array-like, optional\n        The predicted ranks of the column means. If not specified,\n        the columns are assumed to be arranged in order of increasing\n        predicted mean, so the default `predicted_ranks` are\n        :math:`[1, 2, \\dots, n-1, n]`.\n\n    method : {\'auto\', \'asymptotic\', \'exact\'}, optional\n        Selects the method used to calculate the *p*-value. The following\n        options are available.\n\n        * \'auto\': selects between \'exact\' and \'asymptotic\' to\n          achieve reasonably accurate results in reasonable time (default)\n        * \'asymptotic\': compares the standardized test statistic against\n          the normal distribution\n        * \'exact\': computes the exact *p*-value by comparing the observed\n          :math:`L` statistic against those realized by all possible\n          permutations of ranks (under the null hypothesis that each\n          permutation is equally likely)\n\n    Returns\n    -------\n    res : PageTrendTestResult\n        An object containing attributes:\n\n        statistic : float\n            Page\'s :math:`L` test statistic.\n        pvalue : float\n            The associated *p*-value\n        method : {\'asymptotic\', \'exact\'}\n            The method used to compute the *p*-value\n\n    See Also\n    --------\n    rankdata, friedmanchisquare, spearmanr\n\n    Notes\n    -----\n    As noted in [1]_, "the :math:`n` \'treatments\' could just as well represent\n    :math:`n` objects or events or performances or persons or trials ranked."\n    Similarly, the :math:`m` \'subjects\' could equally stand for :math:`m`\n    "groupings by ability or some other control variable, or judges doing\n    the ranking, or random replications of some other sort."\n\n    The procedure for calculating the :math:`L` statistic, adapted from\n    [1]_, is:\n\n    1. "Predetermine with careful logic the appropriate hypotheses\n       concerning the predicted ordering of the experimental results.\n       If no reasonable basis for ordering any treatments is known, the\n       :math:`L` test is not appropriate."\n    2. "As in other experiments, determine at what level of confidence\n       you will reject the null hypothesis that there is no agreement of\n       experimental results with the monotonic hypothesis."\n    3. "Cast the experimental material into a two-way table of :math:`n`\n       columns (treatments, objects ranked, conditions) and :math:`m`\n       rows (subjects, replication groups, levels of control variables)."\n    4. "When experimental observations are recorded, rank them across each\n       row", e.g. ``ranks = scipy.stats.rankdata(data, axis=1)``.\n    5. "Add the ranks in each column", e.g.\n       ``colsums = np.sum(ranks, axis=0)``.\n    6. "Multiply each sum of ranks by the predicted rank for that same\n       column", e.g. ``products = predicted_ranks * colsums``.\n    7. "Sum all such products", e.g. ``L = products.sum()``.\n\n    [1]_ continues by suggesting use of the standardized statistic\n\n    .. math::\n\n        \\chi_L^2 = \\frac{\\left[12L-3mn(n+1)^2\\right]^2}{mn^2(n^2-1)(n+1)}\n\n    "which is distributed approximately as chi-square with 1 degree of\n    freedom. The ordinary use of :math:`\\chi^2` tables would be\n    equivalent to a two-sided test of agreement. If a one-sided test\n    is desired, *as will almost always be the case*, the probability\n    discovered in the chi-square table should be *halved*."\n\n    However, this standardized statistic does not distinguish between the\n    observed values being well correlated with the predicted ranks and being\n    _anti_-correlated with the predicted ranks. Instead, we follow [2]_\n    and calculate the standardized statistic\n\n    .. math::\n\n        \\Lambda = \\frac{L - E_0}{\\sqrt{V_0}},\n\n    where :math:`E_0 = \\frac{1}{4} mn(n+1)^2` and\n    :math:`V_0 = \\frac{1}{144} mn^2(n+1)(n^2-1)`, "which is asymptotically\n    normal under the null hypothesis".\n\n    The *p*-value for ``method=\'exact\'`` is generated by comparing the observed\n    value of :math:`L` against the :math:`L` values generated for all\n    :math:`(n!)^m` possible permutations of ranks. The calculation is performed\n    using the recursive method of [5].\n\n    The *p*-values are not adjusted for the possibility of ties. When\n    ties are present, the reported  ``\'exact\'`` *p*-values may be somewhat\n    larger (i.e. more conservative) than the true *p*-value [2]_. The\n    ``\'asymptotic\'``` *p*-values, however, tend to be smaller (i.e. less\n    conservative) than the ``\'exact\'`` *p*-values.\n\n    References\n    ----------\n    .. [1] Ellis Batten Page, "Ordered hypotheses for multiple treatments:\n       a significant test for linear ranks", *Journal of the American\n       Statistical Association* 58(301), p. 216--230, 1963.\n\n    .. [2] Markus Neuhauser, *Nonparametric Statistical Test: A computational\n       approach*, CRC Press, p. 150--152, 2012.\n\n    .. [3] Statext LLC, "Page\'s L Trend Test - Easy Statistics", *Statext -\n       Statistics Study*, https://www.statext.com/practice/PageTrendTest03.php,\n       Accessed July 12, 2020.\n\n    .. [4] "Page\'s Trend Test", *Wikipedia*, WikimediaFoundation,\n       https://en.wikipedia.org/wiki/Page%27s_trend_test,\n       Accessed July 12, 2020.\n\n    .. [5] Robert E. Odeh, "The exact distribution of Page\'s L-statistic in\n       the two-way layout", *Communications in Statistics - Simulation and\n       Computation*,  6(1), p. 49--61, 1977.\n\n    Examples\n    --------\n    We use the example from [3]_: 10 students are asked to rate three\n    teaching methods - tutorial, lecture, and seminar - on a scale of 1-5,\n    with 1 being the lowest and 5 being the highest. We have decided that\n    a confidence level of 99% is required to reject the null hypothesis in\n    favor of our alternative: that the seminar will have the highest ratings\n    and the tutorial will have the lowest. Initially, the data have been\n    tabulated with each row representing an individual student\'s ratings of\n    the three methods in the following order: tutorial, lecture, seminar.\n\n    >>> table = [[3, 4, 3],\n    ...          [2, 2, 4],\n    ...          [3, 3, 5],\n    ...          [1, 3, 2],\n    ...          [2, 3, 2],\n    ...          [2, 4, 5],\n    ...          [1, 2, 4],\n    ...          [3, 4, 4],\n    ...          [2, 4, 5],\n    ...          [1, 3, 4]]\n\n    Because the tutorial is hypothesized to have the lowest ratings, the\n    column corresponding with tutorial rankings should be first; the seminar\n    is hypothesized to have the highest ratings, so its column should be last.\n    Since the columns are already arranged in this order of increasing\n    predicted mean, we can pass the table directly into `page_trend_test`.\n\n    >>> from scipy.stats import page_trend_test\n    >>> res = page_trend_test(table)\n    >>> res\n    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,\n                        method=\'exact\')\n\n    This *p*-value indicates that there is a 0.1819% chance that\n    the :math:`L` statistic would reach such an extreme value under the null\n    hypothesis. Because 0.1819% is less than 1%, we have evidence to reject\n    the null hypothesis in favor of our alternative at a 99% confidence level.\n\n    The value of the :math:`L` statistic is 133.5. To check this manually,\n    we rank the data such that high scores correspond with high ranks, settling\n    ties with an average rank:\n\n    >>> from scipy.stats import rankdata\n    >>> ranks = rankdata(table, axis=1)\n    >>> ranks\n    array([[1.5, 3. , 1.5],\n           [1.5, 1.5, 3. ],\n           [1.5, 1.5, 3. ],\n           [1. , 3. , 2. ],\n           [1.5, 3. , 1.5],\n           [1. , 2. , 3. ],\n           [1. , 2. , 3. ],\n           [1. , 2.5, 2.5],\n           [1. , 2. , 3. ],\n           [1. , 2. , 3. ]])\n\n    We add the ranks within each column, multiply the sums by the\n    predicted ranks, and sum the products.\n\n    >>> import numpy as np\n    >>> m, n = ranks.shape\n    >>> predicted_ranks = np.arange(1, n+1)\n    >>> L = (predicted_ranks * np.sum(ranks, axis=0)).sum()\n    >>> res.statistic == L\n    True\n\n    As presented in [3]_, the asymptotic approximation of the *p*-value is the\n    survival function of the normal distribution evaluated at the standardized\n    test statistic:\n\n    >>> from scipy.stats import norm\n    >>> E0 = (m*n*(n+1)**2)/4\n    >>> V0 = (m*n**2*(n+1)*(n**2-1))/144\n    >>> Lambda = (L-E0)/np.sqrt(V0)\n    >>> p = norm.sf(Lambda)\n    >>> p\n    0.0012693433690751756\n\n    This does not precisely match the *p*-value reported by `page_trend_test`\n    above. The asymptotic distribution is not very accurate, nor conservative,\n    for :math:`m \\leq 12` and :math:`n \\leq 8`, so `page_trend_test` chose to\n    use ``method=\'exact\'`` based on the dimensions of the table and the\n    recommendations in Page\'s original paper [1]_. To override\n    `page_trend_test`\'s choice, provide the `method` argument.\n\n    >>> res = page_trend_test(table, method="asymptotic")\n    >>> res\n    PageTrendTestResult(statistic=133.5, pvalue=0.0012693433690751756,\n                        method=\'asymptotic\')\n\n    If the data are already ranked, we can pass in the ``ranks`` instead of\n    the ``table`` to save computation time.\n\n    >>> res = page_trend_test(ranks,             # ranks of data\n    ...                       ranked=True,       # data is already ranked\n    ...                       )\n    >>> res\n    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,\n                        method=\'exact\')\n\n    Suppose the raw data had been tabulated in an order different from the\n    order of predicted means, say lecture, seminar, tutorial.\n\n    >>> table = np.asarray(table)[:, [1, 2, 0]]\n\n    Since the arrangement of this table is not consistent with the assumed\n    ordering, we can either rearrange the table or provide the\n    `predicted_ranks`. Remembering that the lecture is predicted\n    to have the middle rank, the seminar the highest, and tutorial the lowest,\n    we pass:\n\n    >>> res = page_trend_test(table,             # data as originally tabulated\n    ...                       predicted_ranks=[2, 3, 1],  # our predicted order\n    ...                       )\n    >>> res\n    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,\n                        method=\'exact\')\n\n    '
    methods = {'asymptotic': _l_p_asymptotic, 'exact': _l_p_exact, 'auto': None}
    if method not in methods:
        raise ValueError(f'`method` must be in {set(methods)}')
    ranks = np.array(data, copy=False)
    if ranks.ndim != 2:
        raise ValueError('`data` must be a 2d array.')
    (m, n) = ranks.shape
    if m < 2 or n < 3:
        raise ValueError("Page's L is only appropriate for data with two or more rows and three or more columns.")
    if np.any(np.isnan(data)):
        raise ValueError('`data` contains NaNs, which cannot be ranked meaningfully')
    if ranked:
        if not (ranks.min() >= 1 and ranks.max() <= ranks.shape[1]):
            raise ValueError('`data` is not properly ranked. Rank the data or pass `ranked=False`.')
    else:
        ranks = scipy.stats.rankdata(data, axis=-1)
    if predicted_ranks is None:
        predicted_ranks = np.arange(1, n + 1)
    else:
        predicted_ranks = np.array(predicted_ranks, copy=False)
        if predicted_ranks.ndim < 1 or (set(predicted_ranks) != set(range(1, n + 1)) or len(predicted_ranks) != n):
            raise ValueError(f'`predicted_ranks` must include each integer from 1 to {n} (the number of columns in `data`) exactly once.')
    if not isinstance(ranked, bool):
        raise TypeError('`ranked` must be boolean.')
    L = _l_vectorized(ranks, predicted_ranks)
    if method == 'auto':
        method = _choose_method(ranks)
    p_fun = methods[method]
    p = p_fun(L, m, n)
    page_result = PageTrendTestResult(statistic=L, pvalue=p, method=method)
    return page_result

def _choose_method(ranks):
    if False:
        i = 10
        return i + 15
    'Choose method for computing p-value automatically'
    (m, n) = ranks.shape
    if n > 8 or (m > 12 and n > 3) or m > 20:
        method = 'asymptotic'
    else:
        method = 'exact'
    return method

def _l_vectorized(ranks, predicted_ranks):
    if False:
        for i in range(10):
            print('nop')
    "Calculate's Page's L statistic for each page of a 3d array"
    colsums = ranks.sum(axis=-2, keepdims=True)
    products = predicted_ranks * colsums
    Ls = products.sum(axis=-1)
    Ls = Ls[0] if Ls.size == 1 else Ls.ravel()
    return Ls

def _l_p_asymptotic(L, m, n):
    if False:
        for i in range(10):
            print('nop')
    "Calculate the p-value of Page's L from the asymptotic distribution"
    E0 = m * n * (n + 1) ** 2 / 4
    V0 = m * n ** 2 * (n + 1) * (n ** 2 - 1) / 144
    Lambda = (L - E0) / np.sqrt(V0)
    p = norm.sf(Lambda)
    return p

def _l_p_exact(L, m, n):
    if False:
        for i in range(10):
            print('nop')
    "Calculate the p-value of Page's L exactly"
    (L, n, k) = (int(L), int(m), int(n))
    _pagel_state.set_k(k)
    return _pagel_state.sf(L, n)

class _PageL:
    """Maintains state between `page_trend_test` executions"""

    def __init__(self):
        if False:
            print('Hello World!')
        'Lightweight initialization'
        self.all_pmfs = {}

    def set_k(self, k):
        if False:
            for i in range(10):
                print('nop')
        'Calculate lower and upper limits of L for single row'
        self.k = k
        (self.a, self.b) = (k * (k + 1) * (k + 2) // 6, k * (k + 1) * (2 * k + 1) // 6)

    def sf(self, l, n):
        if False:
            print('Hello World!')
        "Survival function of Page's L statistic"
        ps = [self.pmf(l, n) for l in range(l, n * self.b + 1)]
        return np.sum(ps)

    def p_l_k_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Relative frequency of each L value over all possible single rows'
        ranks = range(1, self.k + 1)
        rank_perms = np.array(list(permutations(ranks)))
        Ls = (ranks * rank_perms).sum(axis=1)
        counts = np.histogram(Ls, np.arange(self.a - 0.5, self.b + 1.5))[0]
        return counts / math.factorial(self.k)

    def pmf(self, l, n):
        if False:
            for i in range(10):
                print('nop')
        'Recursive function to evaluate p(l, k, n); see [5] Equation 1'
        if n not in self.all_pmfs:
            self.all_pmfs[n] = {}
        if self.k not in self.all_pmfs[n]:
            self.all_pmfs[n][self.k] = {}
        if l in self.all_pmfs[n][self.k]:
            return self.all_pmfs[n][self.k][l]
        if n == 1:
            ps = self.p_l_k_1()
            ls = range(self.a, self.b + 1)
            self.all_pmfs[n][self.k] = {l: p for (l, p) in zip(ls, ps)}
            return self.all_pmfs[n][self.k][l]
        p = 0
        low = max(l - (n - 1) * self.b, self.a)
        high = min(l - (n - 1) * self.a, self.b)
        for t in range(low, high + 1):
            p1 = self.pmf(l - t, n - 1)
            p2 = self.pmf(t, 1)
            p += p1 * p2
        self.all_pmfs[n][self.k][l] = p
        return p
_pagel_state = _PageL()