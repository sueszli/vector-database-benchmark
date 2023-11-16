import math
import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property
from .torch_distribution import TorchDistribution

class AsymmetricLaplace(TorchDistribution):
    """
    Asymmetric version of the :class:`~pyro.distributions.Laplace`
    distribution.

    To the left of ``loc`` this acts like an
    ``-Exponential(1/(asymmetry*scale))``; to the right of ``loc`` this acts
    like an ``Exponential(asymmetry/scale)``. The density is continuous so the
    left and right densities at ``loc`` agree.

    :param loc: Location parameter, i.e. the mode.
    :param scale: Scale parameter = geometric mean of left and right scales.
    :param asymmetry: Square of ratio of left to right scales.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'asymmetry': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, asymmetry, *, validate_args=None):
        if False:
            i = 10
            return i + 15
        (self.loc, self.scale, self.asymmetry) = broadcast_all(loc, scale, asymmetry)
        super().__init__(self.loc.shape, validate_args=validate_args)

    @lazy_property
    def left_scale(self):
        if False:
            print('Hello World!')
        return self.scale * self.asymmetry

    @lazy_property
    def right_scale(self):
        if False:
            for i in range(10):
                print('nop')
        return self.scale / self.asymmetry

    def expand(self, batch_shape, _instance=None):
        if False:
            while True:
                i = 10
        new = self._get_checked_instance(AsymmetricLaplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.asymmetry = self.asymmetry.expand(batch_shape)
        super(AsymmetricLaplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if False:
            return 10
        if self._validate_args:
            self._validate_sample(value)
        z = value - self.loc
        z = -z.abs() / torch.where(z < 0, self.left_scale, self.right_scale)
        return z - (self.left_scale + self.right_scale).log()

    def rsample(self, sample_shape=torch.Size()):
        if False:
            return 10
        shape = self._extended_shape(sample_shape)
        (u, v) = self.loc.new_empty((2,) + shape).exponential_()
        return self.loc - self.left_scale * u + self.right_scale * v

    @property
    def mean(self):
        if False:
            for i in range(10):
                print('nop')
        total_scale = self.left_scale + self.right_scale
        return self.loc + (self.right_scale ** 2 - self.left_scale ** 2) / total_scale

    @property
    def variance(self):
        if False:
            return 10
        left = self.left_scale
        right = self.right_scale
        total = left + right
        p = left / total
        q = right / total
        return p * left ** 2 + q * right ** 2 + p * q * total ** 2

class SoftAsymmetricLaplace(TorchDistribution):
    """
    Soft asymmetric version of the :class:`~pyro.distributions.Laplace`
    distribution.

    This has a smooth (infinitely differentiable) density with two asymmetric
    asymptotically exponential tails, one on the left and one on the right. In
    the limit of ``softness → 0``, this converges in distribution to the
    :class:`AsymmetricLaplace` distribution.

    This is equivalent to the sum of three random variables ``z - u + v`` where::

        z ~ Normal(loc, scale * softness)
        u ~ Exponential(1 / (scale * asymmetry))
        v ~ Exponential(asymetry / scale)

    This is also equivalent the sum of two random variables ``z + a`` where::

        z ~ Normal(loc, scale * softness)
        a ~ AsymmetricLaplace(0, scale, asymmetry)

    :param loc: Location parameter, i.e. the mode.
    :param scale: Scale parameter = geometric mean of left and right scales.
    :param asymmetry: Square of ratio of left to right scales. Defaults to 1.
    :param softness: Scale parameter of the Gaussian smoother. Defaults to 1.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'asymmetry': constraints.positive, 'softness': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, asymmetry=1.0, softness=1.0, *, validate_args=None):
        if False:
            for i in range(10):
                print('nop')
        (self.loc, self.scale, self.asymmetry, self.softness) = broadcast_all(loc, scale, asymmetry, softness)
        super().__init__(self.loc.shape, validate_args=validate_args)

    @lazy_property
    def left_scale(self):
        if False:
            while True:
                i = 10
        return self.scale * self.asymmetry

    @lazy_property
    def right_scale(self):
        if False:
            return 10
        return self.scale / self.asymmetry

    @lazy_property
    def soft_scale(self):
        if False:
            return 10
        return self.scale * self.softness

    def expand(self, batch_shape, _instance=None):
        if False:
            while True:
                i = 10
        new = self._get_checked_instance(AsymmetricLaplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.asymmetry = self.asymmetry.expand(batch_shape)
        new.softness = self.softness.expand(batch_shape)
        super(AsymmetricLaplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if False:
            return 10
        if self._validate_args:
            self._validate_sample(value)
        x = (value - self.loc) / self.scale
        L = self.asymmetry
        R = self.asymmetry.reciprocal()
        S = self.softness
        SS = S * S
        S2 = S * math.sqrt(2)
        Lx = L * x
        Rx = R * x
        return math.log(0.5) + torch.logaddexp((SS / 2 + Lx) / L ** 2 + _logerfc((SS + Lx) / (L * S2)), (SS / 2 - Rx) / R ** 2 + _logerfc((SS - Rx) / (R * S2))) - (L + R).log() - self.scale.log()

    def rsample(self, sample_shape=torch.Size()):
        if False:
            i = 10
            return i + 15
        shape = self._extended_shape(sample_shape)
        z = self.loc.new_empty(shape).normal_()
        (u, v) = self.loc.new_empty((2,) + shape).exponential_()
        return self.loc + self.soft_scale * z - self.left_scale * u + self.right_scale * v

    @property
    def mean(self):
        if False:
            print('Hello World!')
        total_scale = self.left_scale + self.right_scale
        return self.loc + (self.right_scale ** 2 - self.left_scale ** 2) / total_scale

    @property
    def variance(self):
        if False:
            print('Hello World!')
        left = self.left_scale
        right = self.right_scale
        total = left + right
        p = left / total
        q = right / total
        return p * left ** 2 + q * right ** 2 + p * q * total ** 2 + self.soft_scale ** 2

def _logerfc(x):
    if False:
        while True:
            i = 10
    try:
        return torch.logerfc(x)
    except AttributeError:
        return x.double().erfc().log().to(dtype=x.dtype)