import math
import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from pyro.distributions.torch_distribution import TorchDistribution

def _unsafe_standard_stable(alpha, beta, V, W, coords):
    if False:
        for i in range(10):
            print('nop')
    assert V.shape == W.shape
    inv_alpha = alpha.reciprocal()
    half_pi = math.pi / 2
    eps = torch.finfo(V.dtype).eps
    V = V.clamp(min=2 * eps - half_pi, max=half_pi - 2 * eps)
    ha = half_pi * alpha
    b = beta * ha.tan()
    v = b.atan() - ha + alpha * (V + half_pi)
    Z = v.sin() / ((1 + b * b).rsqrt() * V.cos()).pow(inv_alpha) * ((v - V).cos().clamp(min=eps) / W).pow(inv_alpha - 1)
    Z.data[Z.data != Z.data] = 0
    if coords == 'S0':
        return Z - b
    elif coords == 'S':
        return Z
    else:
        raise ValueError('Unknown coords: {}'.format(coords))
RADIUS = 0.01

def _standard_stable(alpha, beta, aux_uniform, aux_exponential, coords):
    if False:
        print('Hello World!')
    '\n    Differentiably transform two random variables::\n\n        aux_uniform ~ Uniform(-pi/2, pi/2)\n        aux_exponential ~ Exponential(1)\n\n    to a standard ``Stable(alpha, beta)`` random variable.\n    '
    with torch.no_grad():
        hole = 1.0
        near_hole = (alpha - hole).abs() <= RADIUS
    if not torch._C._get_tracing_state() and (not near_hole.any()):
        return _unsafe_standard_stable(alpha, beta, aux_uniform, aux_exponential, coords=coords)
    if coords == 'S':
        Z = _standard_stable(alpha, beta, aux_uniform, aux_exponential, 'S0')
        return torch.where(alpha == 1, Z, Z + beta * (math.pi / 2 * alpha).tan())
    aux_uniform_ = aux_uniform.unsqueeze(-1)
    aux_exponential_ = aux_exponential.unsqueeze(-1)
    beta_ = beta.unsqueeze(-1)
    alpha_ = alpha.unsqueeze(-1).expand(alpha.shape + (2,)).contiguous()
    with torch.no_grad():
        (lower, upper) = alpha_.unbind(-1)
        lower.data[near_hole] = hole - RADIUS
        upper.data[near_hole] = hole + RADIUS
        weights = (alpha_ - alpha.unsqueeze(-1)).abs_().mul_(-1 / (2 * RADIUS)).add_(1)
        weights[~near_hole] = 0.5
    pairs = _unsafe_standard_stable(alpha_, beta_, aux_uniform_, aux_exponential_, coords=coords)
    return (pairs * weights).sum(-1)

class Stable(TorchDistribution):
    """
    Levy :math:`\\alpha`-stable distribution. See [1] for a review.

    This uses Nolan's parametrization [2] of the ``loc`` parameter, which is
    required for continuity and differentiability. This corresponds to the
    notation :math:`S^0_\\alpha(\\beta,\\sigma,\\mu_0)` of [1], where
    :math:`\\alpha` = stability, :math:`\\beta` = skew, :math:`\\sigma` = scale,
    and :math:`\\mu_0` = loc. To instead use the S parameterization as in scipy,
    pass ``coords="S"``, but BEWARE this is discontinuous at ``stability=1``
    and has poor geometry for inference.

    This implements a reparametrized sampler :meth:`rsample` , but does not
    implement :meth:`log_prob` . Inference can be performed using either
    likelihood-free algorithms such as
    :class:`~pyro.infer.energy_distance.EnergyDistance`, or reparameterization
    via the :func:`~pyro.poutine.handlers.reparam` handler with one of the
    reparameterizers :class:`~pyro.infer.reparam.stable.LatentStableReparam` ,
    :class:`~pyro.infer.reparam.stable.SymmetricStableReparam` , or
    :class:`~pyro.infer.reparam.stable.StableReparam` e.g.::

        with poutine.reparam(config={"x": StableReparam()}):
            pyro.sample("x", Stable(stability, skew, scale, loc))

    or simply wrap in :class:`~pyro.infer.reparam.strategies.MinimalReparam` or
    :class:`~pyro.infer.reparam.strategies.AutoReparam` , e.g.::

        @MinimalReparam()
        def model():
            ...

    [1] S. Borak, W. Hardle, R. Weron (2005).
        Stable distributions.
        https://edoc.hu-berlin.de/bitstream/handle/18452/4526/8.pdf
    [2] J.P. Nolan (1997).
        Numerical calculation of stable densities and distribution functions.
    [3] Rafal Weron (1996).
        On the Chambers-Mallows-Stuck Method for
        Simulating Skewed Stable Random Variables.
    [4] J.P. Nolan (2017).
        Stable Distributions: Models for Heavy Tailed Data.
        https://edspace.american.edu/jpnolan/wp-content/uploads/sites/1720/2020/09/Chap1.pdf

    :param Tensor stability: Levy stability parameter :math:`\\alpha\\in(0,2]` .
    :param Tensor skew: Skewness :math:`\\beta\\in[-1,1]` .
    :param Tensor scale: Scale :math:`\\sigma > 0` . Defaults to 1.
    :param Tensor loc: Location :math:`\\mu_0` when using Nolan's S0
        parametrization [2], or :math:`\\mu` when using the S parameterization.
        Defaults to 0.
    :param str coords: Either "S0" (default) to use Nolan's continuous S0
        parametrization, or "S" to use the discontinuous parameterization.
    """
    has_rsample = True
    arg_constraints = {'stability': constraints.interval(0, 2), 'skew': constraints.interval(-1, 1), 'scale': constraints.positive, 'loc': constraints.real}
    support = constraints.real

    def __init__(self, stability, skew, scale=1.0, loc=0.0, coords='S0', validate_args=None):
        if False:
            while True:
                i = 10
        assert coords in ('S', 'S0'), coords
        (self.stability, self.skew, self.scale, self.loc) = broadcast_all(stability, skew, scale, loc)
        self.coords = coords
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        if False:
            while True:
                i = 10
        new = self._get_checked_instance(Stable, _instance)
        batch_shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        new.coords = self.coords
        super(Stable, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if False:
            return 10
        raise NotImplementedError('Stable.log_prob() is not implemented')

    def rsample(self, sample_shape=torch.Size()):
        if False:
            print('Hello World!')
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            new_empty = self.stability.new_empty
            aux_uniform = new_empty(shape).uniform_(-math.pi / 2, math.pi / 2)
            aux_exponential = new_empty(shape).exponential_()
        x = _standard_stable(self.stability, self.skew, aux_uniform, aux_exponential, coords=self.coords)
        return self.loc + self.scale * x

    @property
    def mean(self):
        if False:
            return 10
        result = self.loc
        if self.coords == 'S0':
            result = result - self.scale * self.skew * (math.pi / 2 * self.stability).tan()
        return result.masked_fill(self.stability <= 1, math.nan)

    @property
    def variance(self):
        if False:
            return 10
        var = self.scale * self.scale
        return var.mul(2).masked_fill(self.stability < 2, math.inf)