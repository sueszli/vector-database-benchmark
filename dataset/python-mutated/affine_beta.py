import torch
from torch.distributions import constraints
from torch.distributions.transforms import AffineTransform
from .torch import Beta, TransformedDistribution
from .util import broadcast_shape

class AffineBeta(TransformedDistribution):
    """
    Beta distribution scaled by :attr:`scale` and shifted by :attr:`loc`::

        X ~ Beta(concentration1, concentration0)
        f(X) = loc + scale * X
        Y = f(X) ~ AffineBeta(concentration1, concentration0, loc, scale)

    :param concentration1: 1st concentration parameter
        (alpha) for the Beta distribution.
    :type concentration1: float or torch.Tensor
    :param concentration0: 2nd concentration parameter
        (beta) for the Beta distribution.
    :type concentration0: float or torch.Tensor
    :param loc: location parameter.
    :type loc: float or torch.Tensor
    :param scale: scale parameter.
    :type scale: float or torch.Tensor
    """
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive, 'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, concentration1, concentration0, loc, scale, validate_args=None):
        if False:
            return 10
        base_dist = Beta(concentration1, concentration0, validate_args=validate_args)
        super(AffineBeta, self).__init__(base_dist, AffineTransform(loc=loc, scale=scale), validate_args=validate_args)

    @staticmethod
    def infer_shapes(concentration1, concentration0, loc, scale):
        if False:
            return 10
        batch_shape = broadcast_shape(concentration1, concentration0, loc, scale)
        event_shape = torch.Size()
        return (batch_shape, event_shape)

    def expand(self, batch_shape, _instance=None):
        if False:
            i = 10
            return i + 15
        new = self._get_checked_instance(AffineBeta, _instance)
        return super(AffineBeta, self).expand(batch_shape, _instance=new)

    def sample(self, sample_shape=torch.Size()):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates a sample from `Beta` distribution and applies `AffineTransform`.\n        Additionally clamps the output in order to avoid `NaN` and `Inf` values\n        in the gradients.\n        '
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            eps = torch.finfo(x.dtype).eps * self.scale
            x = torch.min(torch.max(x, self.low + eps), self.high - eps)
            return x

    def rsample(self, sample_shape=torch.Size()):
        if False:
            return 10
        '\n        Generates a sample from `Beta` distribution and applies `AffineTransform`.\n        Additionally clamps the output in order to avoid `NaN` and `Inf` values\n        in the gradients.\n        '
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        eps = torch.finfo(x.dtype).eps * self.scale
        x = torch.min(torch.max(x, self.low + eps), self.high - eps)
        return x

    @constraints.dependent_property
    def support(self):
        if False:
            while True:
                i = 10
        return constraints.interval(self.low, self.high)

    @property
    def concentration1(self):
        if False:
            return 10
        return self.base_dist.concentration1

    @property
    def concentration0(self):
        if False:
            for i in range(10):
                print('nop')
        return self.base_dist.concentration0

    @property
    def sample_size(self):
        if False:
            return 10
        return self.concentration1 + self.concentration0

    @property
    def loc(self):
        if False:
            i = 10
            return i + 15
        return torch.as_tensor(self.transforms[0].loc)

    @property
    def scale(self):
        if False:
            print('Hello World!')
        return torch.as_tensor(self.transforms[0].scale)

    @property
    def low(self):
        if False:
            for i in range(10):
                print('nop')
        return self.loc

    @property
    def high(self):
        if False:
            for i in range(10):
                print('nop')
        return self.loc + self.scale

    @property
    def mean(self):
        if False:
            return 10
        return self.loc + self.scale * self.base_dist.mean

    @property
    def variance(self):
        if False:
            for i in range(10):
                print('nop')
        return self.scale.pow(2) * self.base_dist.variance