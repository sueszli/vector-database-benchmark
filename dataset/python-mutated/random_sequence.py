"""
Utilities for generating random numbers, random sequences, and
random selections.
"""
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['powerlaw_sequence', 'zipf_rv', 'cumulative_distribution', 'discrete_sequence', 'random_weighted_sample', 'weighted_choice']

@py_random_state(2)
def powerlaw_sequence(n, exponent=2.0, seed=None):
    if False:
        while True:
            i = 10
    '\n    Return sample sequence of length n from a power law distribution.\n    '
    return [seed.paretovariate(exponent - 1) for i in range(n)]

@py_random_state(2)
def zipf_rv(alpha, xmin=1, seed=None):
    if False:
        i = 10
        return i + 15
    'Returns a random value chosen from the Zipf distribution.\n\n    The return value is an integer drawn from the probability distribution\n\n    .. math::\n\n        p(x)=\\frac{x^{-\\alpha}}{\\zeta(\\alpha, x_{\\min})},\n\n    where $\\zeta(\\alpha, x_{\\min})$ is the Hurwitz zeta function.\n\n    Parameters\n    ----------\n    alpha : float\n      Exponent value of the distribution\n    xmin : int\n      Minimum value\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    x : int\n      Random value from Zipf distribution\n\n    Raises\n    ------\n    ValueError:\n      If xmin < 1 or\n      If alpha <= 1\n\n    Notes\n    -----\n    The rejection algorithm generates random values for a the power-law\n    distribution in uniformly bounded expected time dependent on\n    parameters.  See [1]_ for details on its operation.\n\n    Examples\n    --------\n    >>> nx.utils.zipf_rv(alpha=2, xmin=3, seed=42)\n    8\n\n    References\n    ----------\n    .. [1] Luc Devroye, Non-Uniform Random Variate Generation,\n       Springer-Verlag, New York, 1986.\n    '
    if xmin < 1:
        raise ValueError('xmin < 1')
    if alpha <= 1:
        raise ValueError('a <= 1.0')
    a1 = alpha - 1.0
    b = 2 ** a1
    while True:
        u = 1.0 - seed.random()
        v = seed.random()
        x = int(xmin * u ** (-(1.0 / a1)))
        t = (1.0 + 1.0 / x) ** a1
        if v * x * (t - 1.0) / (b - 1.0) <= t / b:
            break
    return x

def cumulative_distribution(distribution):
    if False:
        print('Hello World!')
    'Returns normalized cumulative distribution from discrete distribution.'
    cdf = [0.0]
    psum = sum(distribution)
    for i in range(len(distribution)):
        cdf.append(cdf[i] + distribution[i] / psum)
    return cdf

@py_random_state(3)
def discrete_sequence(n, distribution=None, cdistribution=None, seed=None):
    if False:
        i = 10
        return i + 15
    '\n    Return sample sequence of length n from a given discrete distribution\n    or discrete cumulative distribution.\n\n    One of the following must be specified.\n\n    distribution = histogram of values, will be normalized\n\n    cdistribution = normalized discrete cumulative distribution\n\n    '
    import bisect
    if cdistribution is not None:
        cdf = cdistribution
    elif distribution is not None:
        cdf = cumulative_distribution(distribution)
    else:
        raise nx.NetworkXError('discrete_sequence: distribution or cdistribution missing')
    inputseq = [seed.random() for i in range(n)]
    seq = [bisect.bisect_left(cdf, s) - 1 for s in inputseq]
    return seq

@py_random_state(2)
def random_weighted_sample(mapping, k, seed=None):
    if False:
        while True:
            i = 10
    'Returns k items without replacement from a weighted sample.\n\n    The input is a dictionary of items with weights as values.\n    '
    if k > len(mapping):
        raise ValueError('sample larger than population')
    sample = set()
    while len(sample) < k:
        sample.add(weighted_choice(mapping, seed))
    return list(sample)

@py_random_state(1)
def weighted_choice(mapping, seed=None):
    if False:
        return 10
    'Returns a single element from a weighted sample.\n\n    The input is a dictionary of items with weights as values.\n    '
    rnd = seed.random() * sum(mapping.values())
    for (k, w) in mapping.items():
        rnd -= w
        if rnd < 0:
            return k