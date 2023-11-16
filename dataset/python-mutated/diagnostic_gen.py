"""
Created on Tue Oct  6 12:42:11 2020

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.effect_size import _noncentrality_chisquare

def test_chisquare_binning(counts, expected, sort_var=None, bins=10, df=None, ordered=False, sort_method='quicksort', alpha_nc=0.05):
    if False:
        i = 10
        return i + 15
    'chisquare gof test with binning of data, Hosmer-Lemeshow type\n\n    ``observed`` and ``expected`` are observation specific and should have\n    observations in rows and choices in columns\n\n    Parameters\n    ----------\n    counts : array_like\n        Observed frequency, i.e. counts for all choices\n    expected : array_like\n        Expected counts or probability. If expected are counts, then they\n        need to sum to the same total count as the sum of observed.\n        If those sums are unequal and all expected values are smaller or equal\n        to 1, then they are interpreted as probabilities and will be rescaled\n        to match counts.\n    sort_var : array_like\n        1-dimensional array for binning. Groups will be formed according to\n        quantiles of the sorted array ``sort_var``, so that group sizes have\n        equal or approximately equal sizes.\n\n    Returns\n    -------\n    Holdertuple instance\n        This instance contains the results of the chisquare test and some\n        information about the data\n\n        - statistic : chisquare statistic of the goodness-of-fit test\n        - pvalue : pvalue of the chisquare test\n        = df : degrees of freedom of the test\n\n    Notes\n    -----\n    Degrees of freedom for Hosmer-Lemeshow tests are given by\n\n    g groups, c choices\n\n    - binary: `df = (g - 2)` for insample,\n         Stata uses `df = g` for outsample\n    - multinomial: `df = (g−2) *(c−1)`, reduces to (g-2) for binary c=2,\n         (Fagerland, Hosmer, Bofin SIM 2008)\n    - ordinal: `df = (g - 2) * (c - 1) + (c - 2)`, reduces to (g-2) for c=2,\n         (Hosmer, ... ?)\n\n    Note: If there are ties in the ``sort_var`` array, then the split of\n    observations into groups will depend on the sort algorithm.\n    '
    observed = np.asarray(counts)
    expected = np.asarray(expected)
    n_observed = counts.sum()
    n_expected = expected.sum()
    if not np.allclose(n_observed, n_expected, atol=1e-13):
        if np.max(expected) < 1 + 1e-13:
            import warnings
            warnings.warn('sum of expected and of observed differ, rescaling ``expected``')
            expected = expected / n_expected * n_observed
        else:
            raise ValueError('total counts of expected and observed differ')
    if sort_var is not None:
        argsort = np.argsort(sort_var, kind=sort_method)
    else:
        argsort = np.arange(observed.shape[0])
    indices = np.array_split(argsort, bins, axis=0)
    freqs = np.array([observed[idx].sum(0) for idx in indices])
    probs = np.array([expected[idx].sum(0) for idx in indices])
    resid_pearson = (freqs - probs) / np.sqrt(probs)
    chi2_stat_groups = ((freqs - probs) ** 2 / probs).sum(1)
    chi2_stat = chi2_stat_groups.sum()
    if df is None:
        (g, c) = freqs.shape
        if ordered is True:
            df = (g - 2) * (c - 1) + (c - 2)
        else:
            df = (g - 2) * (c - 1)
    pvalue = stats.chi2.sf(chi2_stat, df)
    noncentrality = _noncentrality_chisquare(chi2_stat, df, alpha=alpha_nc)
    res = HolderTuple(statistic=chi2_stat, pvalue=pvalue, df=df, freqs=freqs, probs=probs, noncentrality=noncentrality, resid_pearson=resid_pearson, chi2_stat_groups=chi2_stat_groups, indices=indices)
    return res

def prob_larger_ordinal_choice(prob):
    if False:
        return 10
    'probability that observed category is larger than distribution prob\n\n    This is a helper function for Ordinal models, where endog is a 1-dim\n    categorical variable and predicted probabilities are 2-dimensional with\n    observations in rows and choices in columns.\n\n    Parameter\n    ---------\n    prob : array_like\n        Expected probabilities for ordinal choices, e.g. from prediction of\n        an ordinal model with observations in rows and choices in columns.\n\n    Returns\n    -------\n    cdf_mid : ndarray\n        mid cdf, i.e ``P(x < y) + 0.5 P(x=y)``\n    r : ndarray\n        Probability residual ``P(x > y) - P(x < y)`` for all possible choices.\n        Computed as ``r = cdf_mid * 2 - 1``\n\n    References\n    ----------\n    .. [2] Li, Chun, and Bryan E. Shepherd. 2012. “A New Residual for Ordinal\n       Outcomes.” Biometrika 99 (2): 473–80.\n\n    See Also\n    --------\n    `statsmodels.stats.nonparametric.rank_compare_2ordinal`\n\n    '
    prob = np.asarray(prob)
    cdf = prob.cumsum(-1)
    if cdf.ndim == 1:
        cdf_ = np.concatenate(([0], cdf))
    elif cdf.ndim == 2:
        cdf_ = np.concatenate((np.zeros((len(cdf), 1)), cdf), axis=1)
    cdf_mid = (cdf_[..., 1:] + cdf_[..., :-1]) / 2
    r = cdf_mid * 2 - 1
    return (cdf_mid, r)

def prob_larger_2ordinal(probs1, probs2):
    if False:
        for i in range(10):
            print('nop')
    'Stochastically large probability for two ordinal distributions\n\n    Computes Pr(x1 > x2) + 0.5 * Pr(x1 = x2) for two ordered multinomial\n    (ordinal) distributed random variables x1 and x2.\n\n    This is vectorized with choices along last axis.\n    Broadcasting if freq2 is 1-dim also seems to work correctly.\n\n    Returns\n    -------\n    prob1 : float\n        Probability that random draw from distribution 1 is larger than a\n        random draw from distribution 2. Pr(x1 > x2) + 0.5 * Pr(x1 = x2)\n    prob2 : float\n        prob2 = 1 - prob1 = Pr(x1 < x2) + 0.5 * Pr(x1 = x2)\n    '
    freq1 = np.asarray(probs1)
    freq2 = np.asarray(probs2)
    freq1_ = np.concatenate((np.zeros(freq1.shape[:-1] + (1,)), freq1), axis=-1)
    freq2_ = np.concatenate((np.zeros(freq2.shape[:-1] + (1,)), freq2), axis=-1)
    cdf1 = freq1_.cumsum(axis=-1)
    cdf2 = freq2_.cumsum(axis=-1)
    cdfm1 = (cdf1[..., 1:] + cdf1[..., :-1]) / 2
    cdfm2 = (cdf2[..., 1:] + cdf2[..., :-1]) / 2
    prob1 = (cdfm2 * freq1).sum(-1)
    prob2 = (cdfm1 * freq2).sum(-1)
    return (prob1, prob2)

def cov_multinomial(probs):
    if False:
        i = 10
        return i + 15
    'covariance matrix of multinomial distribution\n\n    This is vectorized with choices along last axis.\n\n    cov = diag(probs) - outer(probs, probs)\n\n    '
    k = probs.shape[-1]
    di = np.diag_indices(k, 2)
    cov = probs[..., None] * probs[..., None, :]
    cov *= -1
    cov[..., di[0], di[1]] += probs
    return cov

def var_multinomial(probs):
    if False:
        i = 10
        return i + 15
    'variance of multinomial distribution\n\n    var = probs * (1 - probs)\n\n    '
    var = probs * (1 - probs)
    return var