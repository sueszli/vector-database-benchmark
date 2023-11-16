import math
import torch
import torch.jit
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property
__all__ = ['VonMises']

def _eval_poly(y, coef):
    if False:
        return 10
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result
_I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813]
_I0_COEF_LARGE = [0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281, -0.02057706, 0.02635537, -0.01647633, 0.00392377]
_I1_COEF_SMALL = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532, 0.00032411]
_I1_COEF_LARGE = [0.39894228, -0.03988024, -0.00362018, 0.00163801, -0.01031555, 0.02282967, -0.02895312, 0.01787654, -0.00420059]
_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]

def _log_modified_bessel_fn(x, order=0):
    if False:
        while True:
            i = 10
    '\n    Returns ``log(I_order(x))`` for ``x > 0``,\n    where `order` is either 0 or 1.\n    '
    assert order == 0 or order == 1
    y = x / 3.75
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()
    result = torch.where(x < 3.75, small, large)
    return result

@torch.jit.script_if_tracing
def _rejection_sample(loc, concentration, proposal_r, x):
    if False:
        for i in range(10):
            print('nop')
    done = torch.zeros(x.shape, dtype=torch.bool, device=loc.device)
    while not done.all():
        u = torch.rand((3,) + x.shape, dtype=loc.dtype, device=loc.device)
        (u1, u2, u3) = u.unbind()
        z = torch.cos(math.pi * u1)
        f = (1 + proposal_r * z) / (proposal_r + z)
        c = concentration * (proposal_r - f)
        accept = (c * (2 - c) - u2 > 0) | ((c / u2).log() + 1 - c >= 0)
        if accept.any():
            x = torch.where(accept, (u3 - 0.5).sign() * f.acos(), x)
            done = done | accept
    return (x + math.pi + loc) % (2 * math.pi) - math.pi

class VonMises(Distribution):
    """
    A circular von Mises distribution.

    This implementation uses polar coordinates. The ``loc`` and ``value`` args
    can be any real number (to facilitate unconstrained optimization), but are
    interpreted as angles modulo 2 pi.

    Example::
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = VonMises(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # von Mises distributed with loc=1 and concentration=1
        tensor([1.9777])

    :param torch.Tensor loc: an angle in radians.
    :param torch.Tensor concentration: concentration parameter
    """
    arg_constraints = {'loc': constraints.real, 'concentration': constraints.positive}
    support = constraints.real
    has_rsample = False

    def __init__(self, loc, concentration, validate_args=None):
        if False:
            i = 10
            return i + 15
        (self.loc, self.concentration) = broadcast_all(loc, concentration)
        batch_shape = self.loc.shape
        event_shape = torch.Size()
        tau = 1 + (1 + 4 * self.concentration ** 2).sqrt()
        rho = (tau - (2 * tau).sqrt()) / (2 * self.concentration)
        self._proposal_r = (1 + rho ** 2) / (2 * rho)
        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        if False:
            for i in range(10):
                print('nop')
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.concentration * torch.cos(value - self.loc)
        log_prob = log_prob - math.log(2 * math.pi) - _log_modified_bessel_fn(self.concentration, order=0)
        return log_prob

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        if False:
            i = 10
            return i + 15
        '\n        The sampling algorithm for the von Mises distribution is based on the following paper:\n        Best, D. J., and Nicholas I. Fisher.\n        "Efficient simulation of the von Mises distribution." Applied Statistics (1979): 152-157.\n        '
        shape = self._extended_shape(sample_shape)
        x = torch.empty(shape, dtype=self.loc.dtype, device=self.loc.device)
        return _rejection_sample(self.loc, self.concentration, self._proposal_r, x)

    def expand(self, batch_shape):
        if False:
            return 10
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            loc = self.loc.expand(batch_shape)
            concentration = self.concentration.expand(batch_shape)
            return type(self)(loc, concentration, validate_args=validate_args)

    @property
    def mean(self):
        if False:
            i = 10
            return i + 15
        '\n        The provided mean is the circular one.\n        '
        return self.loc

    @property
    def mode(self):
        if False:
            return 10
        return self.loc

    @lazy_property
    def variance(self):
        if False:
            print('Hello World!')
        '\n        The provided variance is the circular one.\n        '
        return 1 - (_log_modified_bessel_fn(self.concentration, order=1) - _log_modified_bessel_fn(self.concentration, order=0)).exp()