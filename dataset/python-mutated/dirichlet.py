import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
__all__ = ['Dirichlet']

def _Dirichlet_backward(x, concentration, grad_output):
    if False:
        return 10
    total = concentration.sum(-1, True).expand_as(concentration)
    grad = torch._dirichlet_grad(x, concentration, total)
    return grad * (grad_output - (x * grad_output).sum(-1, True))

class _Dirichlet(Function):

    @staticmethod
    def forward(ctx, concentration):
        if False:
            while True:
                i = 10
        x = torch._sample_dirichlet(concentration)
        ctx.save_for_backward(x, concentration)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if False:
            print('Hello World!')
        (x, concentration) = ctx.saved_tensors
        return _Dirichlet_backward(x, concentration, grad_output)

class Dirichlet(ExponentialFamily):
    """
    Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentration [0.5, 0.5]
        tensor([ 0.1046,  0.8954])

    Args:
        concentration (Tensor): concentration parameter of the distribution
            (often referred to as alpha)
    """
    arg_constraints = {'concentration': constraints.independent(constraints.positive, 1)}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, concentration, validate_args=None):
        if False:
            print('Hello World!')
        if concentration.dim() < 1:
            raise ValueError('`concentration` parameter must be at least one-dimensional.')
        self.concentration = concentration
        (batch_shape, event_shape) = (concentration.shape[:-1], concentration.shape[-1:])
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        if False:
            while True:
                i = 10
        new = self._get_checked_instance(Dirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape + self.event_shape)
        super(Dirichlet, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=()):
        if False:
            return 10
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        return _Dirichlet.apply(concentration)

    def log_prob(self, value):
        if False:
            i = 10
            return i + 15
        if self._validate_args:
            self._validate_sample(value)
        return torch.xlogy(self.concentration - 1.0, value).sum(-1) + torch.lgamma(self.concentration.sum(-1)) - torch.lgamma(self.concentration).sum(-1)

    @property
    def mean(self):
        if False:
            return 10
        return self.concentration / self.concentration.sum(-1, True)

    @property
    def mode(self):
        if False:
            return 10
        concentrationm1 = (self.concentration - 1).clamp(min=0.0)
        mode = concentrationm1 / concentrationm1.sum(-1, True)
        mask = (self.concentration < 1).all(axis=-1)
        mode[mask] = torch.nn.functional.one_hot(mode[mask].argmax(axis=-1), concentrationm1.shape[-1]).to(mode)
        return mode

    @property
    def variance(self):
        if False:
            while True:
                i = 10
        con0 = self.concentration.sum(-1, True)
        return self.concentration * (con0 - self.concentration) / (con0.pow(2) * (con0 + 1))

    def entropy(self):
        if False:
            for i in range(10):
                print('nop')
        k = self.concentration.size(-1)
        a0 = self.concentration.sum(-1)
        return torch.lgamma(self.concentration).sum(-1) - torch.lgamma(a0) - (k - a0) * torch.digamma(a0) - ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1)

    @property
    def _natural_params(self):
        if False:
            return 10
        return (self.concentration,)

    def _log_normalizer(self, x):
        if False:
            return 10
        return x.lgamma().sum(-1) - torch.lgamma(x.sum(-1))