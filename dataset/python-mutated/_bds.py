"""
BDS test for IID time series

References
----------

Broock, W. A., J. A. Scheinkman, W. D. Dechert, and B. LeBaron. 1996.
"A Test for Independence Based on the Correlation Dimension."
Econometric Reviews 15 (3): 197-235.

Kanzler, Ludwig. 1999.
"Very Fast and Correctly Sized Estimation of the BDS Statistic".
SSRN Scholarly Paper ID 151669. Rochester, NY: Social Science Research Network.

LeBaron, Blake. 1997.
"A Fast Algorithm for the BDS Statistic."
Studies in Nonlinear Dynamics & Econometrics 2 (2) (January 1).
"""
import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like

def distance_indicators(x, epsilon=None, distance=1.5):
    if False:
        while True:
            i = 10
    '\n    Calculate all pairwise threshold distance indicators for a time series\n\n    Parameters\n    ----------\n    x : 1d array\n        observations of time series for which heaviside distance indicators\n        are calculated\n    epsilon : scalar, optional\n        the threshold distance to use in calculating the heaviside indicators\n    distance : scalar, optional\n        if epsilon is omitted, specifies the distance multiplier to use when\n        computing it\n\n    Returns\n    -------\n    indicators : 2d array\n        matrix of distance threshold indicators\n\n    Notes\n    -----\n    Since this can be a very large matrix, use np.int8 to save some space.\n    '
    x = array_like(x, 'x')
    if epsilon is not None and epsilon <= 0:
        raise ValueError('Threshold distance must be positive if specified. Got epsilon of %f' % epsilon)
    if distance <= 0:
        raise ValueError('Threshold distance must be positive. Got distance multiplier %f' % distance)
    if epsilon is None:
        epsilon = distance * x.std(ddof=1)
    return np.abs(x[:, None] - x) < epsilon

def correlation_sum(indicators, embedding_dim):
    if False:
        i = 10
        return i + 15
    '\n    Calculate a correlation sum\n\n    Useful as an estimator of a correlation integral\n\n    Parameters\n    ----------\n    indicators : ndarray\n        2d array of distance threshold indicators\n    embedding_dim : int\n        embedding dimension\n\n    Returns\n    -------\n    corrsum : float\n        Correlation sum\n    indicators_joint\n        matrix of joint-distance-threshold indicators\n    '
    if not indicators.ndim == 2:
        raise ValueError('Indicators must be a matrix')
    if not indicators.shape[0] == indicators.shape[1]:
        raise ValueError('Indicator matrix must be symmetric (square)')
    if embedding_dim == 1:
        indicators_joint = indicators
    else:
        (corrsum, indicators) = correlation_sum(indicators, embedding_dim - 1)
        indicators_joint = indicators[1:, 1:] * indicators[:-1, :-1]
    nobs = len(indicators_joint)
    corrsum = np.mean(indicators_joint[np.triu_indices(nobs, 1)])
    return (corrsum, indicators_joint)

def correlation_sums(indicators, max_dim):
    if False:
        print('Hello World!')
    '\n    Calculate all correlation sums for embedding dimensions 1:max_dim\n\n    Parameters\n    ----------\n    indicators : 2d array\n        matrix of distance threshold indicators\n    max_dim : int\n        maximum embedding dimension\n\n    Returns\n    -------\n    corrsums : ndarray\n        Correlation sums\n    '
    corrsums = np.zeros((1, max_dim))
    (corrsums[0, 0], indicators) = correlation_sum(indicators, 1)
    for i in range(1, max_dim):
        (corrsums[0, i], indicators) = correlation_sum(indicators, 2)
    return corrsums

def _var(indicators, max_dim):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the variance of a BDS effect\n\n    Parameters\n    ----------\n    indicators : ndarray\n        2d array of distance threshold indicators\n    max_dim : int\n        maximum embedding dimension\n\n    Returns\n    -------\n    variances : float\n        Variance of BDS effect\n    '
    nobs = len(indicators)
    (corrsum_1dim, _) = correlation_sum(indicators, 1)
    k = ((indicators.sum(1) ** 2).sum() - 3 * indicators.sum() + 2 * nobs) / (nobs * (nobs - 1) * (nobs - 2))
    variances = np.zeros((1, max_dim - 1))
    for embedding_dim in range(2, max_dim + 1):
        tmp = 0
        for j in range(1, embedding_dim):
            tmp += k ** (embedding_dim - j) * corrsum_1dim ** (2 * j)
        variances[0, embedding_dim - 2] = 4 * (k ** embedding_dim + 2 * tmp + (embedding_dim - 1) ** 2 * corrsum_1dim ** (2 * embedding_dim) - embedding_dim ** 2 * k * corrsum_1dim ** (2 * embedding_dim - 2))
    return (variances, k)

def bds(x, max_dim=2, epsilon=None, distance=1.5):
    if False:
        return 10
    '\n    BDS Test Statistic for Independence of a Time Series\n\n    Parameters\n    ----------\n    x : ndarray\n        Observations of time series for which bds statistics is calculated.\n    max_dim : int\n        The maximum embedding dimension.\n    epsilon : {float, None}, optional\n        The threshold distance to use in calculating the correlation sum.\n    distance : float, optional\n        Specifies the distance multiplier to use when computing the test\n        statistic if epsilon is omitted.\n\n    Returns\n    -------\n    bds_stat : float\n        The BDS statistic.\n    pvalue : float\n        The p-values associated with the BDS statistic.\n\n    Notes\n    -----\n    The null hypothesis of the test statistic is for an independent and\n    identically distributed (i.i.d.) time series, and an unspecified\n    alternative hypothesis.\n\n    This test is often used as a residual diagnostic.\n\n    The calculation involves matrices of size (nobs, nobs), so this test\n    will not work with very long datasets.\n\n    Implementation conditions on the first m-1 initial values, which are\n    required to calculate the m-histories:\n    x_t^m = (x_t, x_{t-1}, ... x_{t-(m-1)})\n    '
    x = array_like(x, 'x', ndim=1)
    nobs_full = len(x)
    if max_dim < 2 or max_dim >= nobs_full:
        raise ValueError('Maximum embedding dimension must be in the range [2,len(x)-1]. Got %d.' % max_dim)
    indicators = distance_indicators(x, epsilon, distance)
    corrsum_mdims = correlation_sums(indicators, max_dim)
    (variances, k) = _var(indicators, max_dim)
    stddevs = np.sqrt(variances)
    bds_stats = np.zeros((1, max_dim - 1))
    pvalues = np.zeros((1, max_dim - 1))
    for embedding_dim in range(2, max_dim + 1):
        ninitial = embedding_dim - 1
        nobs = nobs_full - ninitial
        (corrsum_1dim, _) = correlation_sum(indicators[ninitial:, ninitial:], 1)
        corrsum_mdim = corrsum_mdims[0, embedding_dim - 1]
        effect = corrsum_mdim - corrsum_1dim ** embedding_dim
        sd = stddevs[0, embedding_dim - 2]
        bds_stats[0, embedding_dim - 2] = np.sqrt(nobs) * effect / sd
        pvalue = 2 * stats.norm.sf(np.abs(bds_stats[0, embedding_dim - 2]))
        pvalues[0, embedding_dim - 2] = pvalue
    return (np.squeeze(bds_stats), np.squeeze(pvalues))