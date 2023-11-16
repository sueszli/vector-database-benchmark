import torch
from torch.distributions import constraints
from torch.distributions.gamma import Gamma
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import PowerTransform
__all__ = ['InverseGamma']

class InverseGamma(TransformedDistribution):
    """
    Creates an inverse gamma distribution parameterized by :attr:`concentration` and :attr:`rate`
    where::

        X ~ Gamma(concentration, rate)
        Y = 1 / X ~ InverseGamma(concentration, rate)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = InverseGamma(torch.tensor([2.0]), torch.tensor([3.0]))
        >>> m.sample()
        tensor([ 1.2953])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """
    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, concentration, rate, validate_args=None):
        if False:
            while True:
                i = 10
        base_dist = Gamma(concentration, rate, validate_args=validate_args)
        neg_one = -base_dist.rate.new_ones(())
        super().__init__(base_dist, PowerTransform(neg_one), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        if False:
            while True:
                i = 10
        new = self._get_checked_instance(InverseGamma, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def concentration(self):
        if False:
            print('Hello World!')
        return self.base_dist.concentration

    @property
    def rate(self):
        if False:
            i = 10
            return i + 15
        return self.base_dist.rate

    @property
    def mean(self):
        if False:
            i = 10
            return i + 15
        result = self.rate / (self.concentration - 1)
        return torch.where(self.concentration > 1, result, torch.inf)

    @property
    def mode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rate / (self.concentration + 1)

    @property
    def variance(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.rate.square() / ((self.concentration - 1).square() * (self.concentration - 2))
        return torch.where(self.concentration > 2, result, torch.inf)

    def entropy(self):
        if False:
            for i in range(10):
                print('nop')
        return self.concentration + self.rate.log() + self.concentration.lgamma() - (1 + self.concentration) * self.concentration.digamma()