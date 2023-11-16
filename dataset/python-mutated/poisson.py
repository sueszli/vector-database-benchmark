from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
__all__ = ['Poisson']

class Poisson(ExponentialFamily):
    """
    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \\mathrm{rate}^k \\frac{e^{-\\mathrm{rate}}}{k!}

    Example::

        >>> # xdoctest: +SKIP("poisson_cpu not implemented for 'Long'")
        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
        tensor([ 3.])

    Args:
        rate (Number, Tensor): the rate parameter
    """
    arg_constraints = {'rate': constraints.nonnegative}
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        if False:
            i = 10
            return i + 15
        return self.rate

    @property
    def mode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rate.floor()

    @property
    def variance(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rate

    def __init__(self, rate, validate_args=None):
        if False:
            i = 10
            return i + 15
        (self.rate,) = broadcast_all(rate)
        if isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        if False:
            while True:
                i = 10
        new = self._get_checked_instance(Poisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Poisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        if False:
            return 10
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))

    def log_prob(self, value):
        if False:
            while True:
                i = 10
        if self._validate_args:
            self._validate_sample(value)
        (rate, value) = broadcast_all(self.rate, value)
        return value.xlogy(rate) - rate - (value + 1).lgamma()

    @property
    def _natural_params(self):
        if False:
            i = 10
            return i + 15
        return (torch.log(self.rate),)

    def _log_normalizer(self, x):
        if False:
            for i in range(10):
                print('nop')
        return torch.exp(x)