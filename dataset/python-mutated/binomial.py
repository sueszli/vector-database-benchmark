import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property, logits_to_probs, probs_to_logits
__all__ = ['Binomial']

def _clamp_by_zero(x):
    if False:
        return 10
    return (x.clamp(min=0) + x - x.clamp(max=0)) / 2

class Binomial(Distribution):
    """
    Creates a Binomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be
    broadcastable with :attr:`probs`/:attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
        tensor([   0.,   22.,   71.,  100.])

        >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
        >>> x = m.sample()
        tensor([[ 4.,  5.],
                [ 7.,  6.]])

    Args:
        total_count (int or Tensor): number of Bernoulli trials
        probs (Tensor): Event probabilities
        logits (Tensor): Event log-odds
    """
    arg_constraints = {'total_count': constraints.nonnegative_integer, 'probs': constraints.unit_interval, 'logits': constraints.real}
    has_enumerate_support = True

    def __init__(self, total_count=1, probs=None, logits=None, validate_args=None):
        if False:
            print('Hello World!')
        if (probs is None) == (logits is None):
            raise ValueError('Either `probs` or `logits` must be specified, but not both.')
        if probs is not None:
            (self.total_count, self.probs) = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.probs)
        else:
            (self.total_count, self.logits) = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)
        self._param = self.probs if probs is not None else self.logits
        batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        if False:
            i = 10
            return i + 15
        new = self._get_checked_instance(Binomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Binomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        if False:
            return 10
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        if False:
            while True:
                i = 10
        return constraints.integer_interval(0, self.total_count)

    @property
    def mean(self):
        if False:
            while True:
                i = 10
        return self.total_count * self.probs

    @property
    def mode(self):
        if False:
            i = 10
            return i + 15
        return ((self.total_count + 1) * self.probs).floor().clamp(max=self.total_count)

    @property
    def variance(self):
        if False:
            print('Hello World!')
        return self.total_count * self.probs * (1 - self.probs)

    @lazy_property
    def logits(self):
        if False:
            for i in range(10):
                print('nop')
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        if False:
            return 10
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        if False:
            for i in range(10):
                print('nop')
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        if False:
            return 10
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.binomial(self.total_count.expand(shape), self.probs.expand(shape))

    def log_prob(self, value):
        if False:
            while True:
                i = 10
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        normalize_term = self.total_count * _clamp_by_zero(self.logits) + self.total_count * torch.log1p(torch.exp(-torch.abs(self.logits))) - log_factorial_n
        return value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term

    def entropy(self):
        if False:
            print('Hello World!')
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError('Inhomogeneous total count not supported by `entropy`.')
        log_prob = self.log_prob(self.enumerate_support(False))
        return -(torch.exp(log_prob) * log_prob).sum(0)

    def enumerate_support(self, expand=True):
        if False:
            print('Hello World!')
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError('Inhomogeneous total count not supported by `enumerate_support`.')
        values = torch.arange(1 + total_count, dtype=self._param.dtype, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values