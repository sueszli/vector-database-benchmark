import torch
from torch.distributions.distribution import Distribution
__all__ = ['ExponentialFamily']

class ExponentialFamily(Distribution):
    """
    ExponentialFamily is the abstract base class for probability distributions belonging to an
    exponential family, whose probability mass/density function has the form is defined below

    .. math::

        p_{F}(x; \\theta) = \\exp(\\langle t(x), \\theta\\rangle - F(\\theta) + k(x))

    where :math:`\\theta` denotes the natural parameters, :math:`t(x)` denotes the sufficient statistic,
    :math:`F(\\theta)` is the log normalizer function for a given family and :math:`k(x)` is the carrier
    measure.

    Note:
        This class is an intermediary between the `Distribution` class and distributions which belong
        to an exponential family mainly to check the correctness of the `.entropy()` and analytic KL
        divergence methods. We use this class to compute the entropy and KL divergence using the AD
        framework and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and
        Cross-entropies of Exponential Families).
    """

    @property
    def _natural_params(self):
        if False:
            return 10
        '\n        Abstract method for natural parameters. Returns a tuple of Tensors based\n        on the distribution\n        '
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        if False:
            i = 10
            return i + 15
        '\n        Abstract method for log normalizer function. Returns a log normalizer based on\n        the distribution and input\n        '
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self):
        if False:
            while True:
                i = 10
        '\n        Abstract method for expected carrier measure, which is required for computing\n        entropy.\n        '
        raise NotImplementedError

    def entropy(self):
        if False:
            return 10
        '\n        Method to compute the entropy using Bregman divergence of the log normalizer.\n        '
        result = -self._mean_carrier_measure
        nparams = [p.detach().requires_grad_() for p in self._natural_params]
        lg_normal = self._log_normalizer(*nparams)
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        result += lg_normal
        for (np, g) in zip(nparams, gradients):
            result -= (np * g).reshape(self._batch_shape + (-1,)).sum(-1)
        return result