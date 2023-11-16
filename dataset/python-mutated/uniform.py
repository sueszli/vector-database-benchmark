from numbers import Number
import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
__all__ = ['Uniform']

class Uniform(Distribution):
    """
    Generates uniformly distributed random samples from the half-open interval
    ``[low, high)``.

    Example::

        >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
        >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
        >>> # xdoctest: +SKIP
        tensor([ 2.3418])

    Args:
        low (float or Tensor): lower range (inclusive).
        high (float or Tensor): upper range (exclusive).
    """
    arg_constraints = {'low': constraints.dependent(is_discrete=False, event_dim=0), 'high': constraints.dependent(is_discrete=False, event_dim=0)}
    has_rsample = True

    @property
    def mean(self):
        if False:
            return 10
        return (self.high + self.low) / 2

    @property
    def mode(self):
        if False:
            return 10
        return nan * self.high

    @property
    def stddev(self):
        if False:
            print('Hello World!')
        return (self.high - self.low) / 12 ** 0.5

    @property
    def variance(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.high - self.low).pow(2) / 12

    def __init__(self, low, high, validate_args=None):
        if False:
            i = 10
            return i + 15
        (self.low, self.high) = broadcast_all(low, high)
        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self._validate_args and (not torch.lt(self.low, self.high).all()):
            raise ValueError('Uniform is not defined when low>= high')

    def expand(self, batch_shape, _instance=None):
        if False:
            return 10
        new = self._get_checked_instance(Uniform, _instance)
        batch_shape = torch.Size(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(Uniform, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        if False:
            for i in range(10):
                print('nop')
        return constraints.interval(self.low, self.high)

    def rsample(self, sample_shape=torch.Size()):
        if False:
            i = 10
            return i + 15
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        if False:
            i = 10
            return i + 15
        if self._validate_args:
            self._validate_sample(value)
        lb = self.low.le(value).type_as(self.low)
        ub = self.high.gt(value).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)

    def cdf(self, value):
        if False:
            while True:
                i = 10
        if self._validate_args:
            self._validate_sample(value)
        result = (value - self.low) / (self.high - self.low)
        return result.clamp(min=0, max=1)

    def icdf(self, value):
        if False:
            return 10
        result = value * (self.high - self.low) + self.low
        return result

    def entropy(self):
        if False:
            while True:
                i = 10
        return torch.log(self.high - self.low)