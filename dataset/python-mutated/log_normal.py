from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform
__all__ = ['LogNormal']

class LogNormal(TransformedDistribution):
    """
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        if False:
            for i in range(10):
                print('nop')
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        if False:
            i = 10
            return i + 15
        new = self._get_checked_instance(LogNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        if False:
            while True:
                i = 10
        return self.base_dist.loc

    @property
    def scale(self):
        if False:
            for i in range(10):
                print('nop')
        return self.base_dist.scale

    @property
    def mean(self):
        if False:
            while True:
                i = 10
        return (self.loc + self.scale.pow(2) / 2).exp()

    @property
    def mode(self):
        if False:
            while True:
                i = 10
        return (self.loc - self.scale.square()).exp()

    @property
    def variance(self):
        if False:
            print('Hello World!')
        scale_sq = self.scale.pow(2)
        return scale_sq.expm1() * (2 * self.loc + scale_sq).exp()

    def entropy(self):
        if False:
            i = 10
            return i + 15
        return self.base_dist.entropy() + self.loc