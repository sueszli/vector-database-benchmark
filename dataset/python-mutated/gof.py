"""
Goodness of Fit Testing
-----------------------

This module implements goodness of fit tests for checking agreement between
distributions' ``.sample()`` and ``.log_prob()`` methods. The main functions
return a goodness of fit p-value ``gof`` which for good data should be
``Uniform(0,1)`` distributed and for bad data should be close to zero. To use
this returned number in tests, set a global variable ``TEST_FAILURE_RATE`` to
something smaller than 1 / number of tests in your suite, then in each test
assert ``gof > TEST_FAILURE_RATE``. For example::

    TEST_FAILURE_RATE = 1 / 20  # For 1 in 20 chance of spurious failure.

    def test_my_distribution():
        d = MyDistribution()
        samples = d.sample([10000])
        probs = d.log_prob(samples).exp()
        gof = auto_goodness_of_fit(samples, probs)
        assert gof > TEST_FAILURE_RATE

This module is a port of the
`goftests <https://github.com/posterior/goftests>`_ library.
"""
import math
import warnings
import torch
from .special import chi2sf
HISTOGRAM_WIDTH = 60

class InvalidTest(ValueError):
    pass

def print_histogram(probs, counts):
    if False:
        return 10
    max_count = int(max(counts))
    print('{: >8} {: >8}'.format('Prob', 'Count'))
    for (prob, count) in sorted(zip(probs, counts), reverse=True):
        width = int(round(HISTOGRAM_WIDTH * int(count) / max_count))
        print('{: >8.3f} {: >8d} {}'.format(prob, count, '-' * width))

@torch.no_grad()
def multinomial_goodness_of_fit(probs, counts, *, total_count=None, plot=False):
    if False:
        return 10
    "\n    Pearson's chi^2 test, on possibly truncated data.\n    https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test\n\n    :param torch.Tensor probs: Vector of probabilities.\n    :param torch.Tensor counts: Vector of counts.\n    :param int total_count: Optional total count in case data is truncated,\n        otherwise None.\n    :param bool plot: Whether to print a histogram. Defaults to False.\n    :returns: p-value of truncated multinomial sample.\n    :rtype: float\n    "
    assert probs.dim() == 1
    assert probs.shape == counts.shape
    if total_count is None:
        truncated = False
        total_count = int(counts.sum())
    else:
        truncated = True
        assert total_count >= counts.sum()
    if plot:
        print_histogram(probs, counts)
    chi_squared = 0
    dof = 0
    for (p, c) in zip(probs.tolist(), counts.tolist()):
        if abs(p - 1) < 1e-08:
            return 1 if c == total_count else 0
        assert p < 1, f'bad probability: {p:g}'
        if p > 0:
            mean = total_count * p
            variance = total_count * p * (1 - p)
            if not variance > 1:
                raise InvalidTest('Goodness of fit is inaccurate; use more samples')
            chi_squared += (c - mean) ** 2 / variance
            dof += 1
        else:
            warnings.warn('Zero probability in goodness-of-fit test')
            if c > 0:
                return math.inf
    if not truncated:
        dof -= 1
    survival = chi2sf(chi_squared, dof)
    return survival

@torch.no_grad()
def unif01_goodness_of_fit(samples, *, plot=False):
    if False:
        while True:
            i = 10
    "\n    Bin uniformly distributed samples and apply Pearson's chi^2 test.\n\n    :param torch.Tensor samples: A vector of real-valued samples from a\n        candidate distribution that should be Uniform(0, 1)-distributed.\n    :param bool plot: Whether to print a histogram. Defaults to False.\n    :returns: Goodness of fit, as a p-value.\n    :rtype: float\n    "
    assert samples.min() >= 0
    assert samples.max() <= 1
    bin_count = int(round(len(samples) ** 0.333))
    if bin_count < 7:
        raise InvalidTest('imprecise test, use more samples')
    probs = torch.ones(bin_count) / bin_count
    binned = samples.mul(bin_count).long().clamp(min=0, max=bin_count - 1)
    counts = torch.zeros(bin_count)
    counts.scatter_add_(0, binned, torch.ones(binned.shape))
    return multinomial_goodness_of_fit(probs, counts, plot=plot)

@torch.no_grad()
def exp_goodness_of_fit(samples, plot=False):
    if False:
        return 10
    "\n    Transform exponentially distribued samples to Uniform(0,1) distribution and\n    assess goodness of fit via binned Pearson's chi^2 test.\n\n    :param torch.Tensor samples: A vector of real-valued samples from a\n        candidate distribution that should be Exponential(1)-distributed.\n    :param bool plot: Whether to print a histogram. Defaults to False.\n    :returns: Goodness of fit, as a p-value.\n    :rtype: float\n    "
    unif01_samples = samples.neg().exp()
    return unif01_goodness_of_fit(unif01_samples, plot=plot)

@torch.no_grad()
def density_goodness_of_fit(samples, probs, plot=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Transform arbitrary continuous samples to Uniform(0,1) distribution and\n    assess goodness of fit via binned Pearson's chi^2 test.\n\n    :param torch.Tensor samples: A vector list of real-valued samples from a\n        distribution.\n    :param torch.Tensor probs: A vector of probability densities evaluated at\n        those samples.\n    :param bool plot: Whether to print a histogram. Defaults to False.\n    :returns: Goodness of fit, as a p-value.\n    :rtype: float\n    "
    assert samples.shape == probs.shape
    if len(samples) <= 100:
        raise InvalidTest('imprecision; use more samples')
    (samples, index) = samples.sort(0)
    probs = probs[index]
    gaps = samples[1:] - samples[:-1]
    sparsity = 1 / probs
    sparsity = 0.5 * (sparsity[1:] + sparsity[:-1])
    density = len(samples) / sparsity
    exp_samples = density * gaps
    return exp_goodness_of_fit(exp_samples, plot=plot)

def volume_of_sphere(dim, radius):
    if False:
        return 10
    return radius ** dim * math.pi ** (0.5 * dim) / math.gamma(0.5 * dim + 1)

def get_nearest_neighbor_distances(samples):
    if False:
        while True:
            i = 10
    try:
        from scipy.spatial import cKDTree
        samples = samples.cpu().numpy()
        (distances, indices) = cKDTree(samples).query(samples, k=2)
        return torch.from_numpy(distances[:, 1])
    except ImportError:
        (distances, indices) = torch.cdist(samples, samples).kthvalue(k=2)
        return distances

@torch.no_grad()
def vector_density_goodness_of_fit(samples, probs, *, dim=None, plot=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Transform arbitrary multivariate continuous samples to Univariate(0,1)\n    distribution via nearest neighbor distribution [1,2,3] and assess goodness\n    of fit via binned Pearson\'s chi^2 test.\n\n    [1] Peter J. Bickel and Leo Breiman (1983)\n        "Sums of Functions of Nearest Neighbor Distances, Moment Bounds, Limit\n        Theorems and a Goodness of Fit Test"\n        https://projecteuclid.org/download/pdf_1/euclid.aop/1176993668\n    [2] Mike Williams (2010)\n        "How good are your fits? Unbinned multivariate goodness-of-fit tests in\n        high energy physics."\n        https://arxiv.org/abs/1006.3019\n    [3] Nearest Neighbour Distribution\n        https://en.wikipedia.org/wiki/Nearest_neighbour_distribution\n\n    :param torch.Tensor samples: A tensor of real-vector-valued samples from a\n        distribution.\n    :param torch.Tensor probs: A vector of probability densities evaluated at\n        those samples.\n    :param int dim: Optional dimension of the submanifold on which data lie.\n        Defaults to ``samples.shape[-1]``.\n    :param bool plot: Whether to print a histogram. Defaults to False.\n    :returns: Goodness of fit, as a p-value.\n    :rtype: float\n    '
    assert samples.shape and len(samples)
    assert probs.shape == samples.shape[:1]
    if dim is None:
        dim = samples.shape[-1]
    assert dim
    if len(samples) <= 1000 * dim:
        raise InvalidTest('imprecision; use more samples')
    radii = get_nearest_neighbor_distances(samples)
    density = len(samples) * probs
    volume = volume_of_sphere(dim, radii)
    exp_samples = density * volume
    return exp_goodness_of_fit(exp_samples, plot=plot)

@torch.no_grad()
def auto_goodness_of_fit(samples, probs, *, dim=None, plot=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Dispatch on sample dimension and delegate to either\n    :func:`density_goodness_of_fit` or :func:`vector_density_goodness_of_fit`.\n\n    :param torch.Tensor samples: A tensor of samples stacked on their leftmost\n        dimension.\n    :param torch.Tensor probs: A vector of probabilities evaluated at those\n        samples.\n    :param int dim: Optional manifold dimension, defaults to\n        ``samples.shape[1:].numel()``.\n    :param bool plot: Whether to print a histogram. Defaults to False.\n    '
    assert samples.shape and samples.shape[0]
    assert probs.shape == samples.shape[:1]
    samples = samples.reshape(samples.shape[0], -1)
    ambient_dim = samples.shape[1:].numel()
    if dim is None:
        dim = ambient_dim
    if ambient_dim == 0:
        return 1.0
    if ambient_dim == 1:
        samples = samples.reshape(-1)
        return density_goodness_of_fit(samples, probs, plot=plot)
    return vector_density_goodness_of_fit(samples, probs, dim=dim, plot=plot)