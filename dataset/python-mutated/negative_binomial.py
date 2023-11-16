import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property, logits_to_probs, probs_to_logits
__all__ = ['NegativeBinomial']

class NegativeBinomial(Distribution):
    """
    Creates a Negative Binomial distribution, i.e. distribution
    of the number of successful independent and identical Bernoulli trials
    before :attr:`total_count` failures are achieved. The probability
    of success of each Bernoulli trial is :attr:`probs`.

    Args:
        total_count (float or Tensor): non-negative number of negative Bernoulli
            trials to stop, although the distribution is still valid for real
            valued count
        probs (Tensor): Event probabilities of success in the half open interval [0, 1)
        logits (Tensor): Event log-odds for probabilities of success
    """
    arg_constraints = {'total_count': constraints.greater_than_eq(0), 'probs': constraints.half_open_interval(0.0, 1.0), 'logits': constraints.real}
    support = constraints.nonnegative_integer

    def __init__(self, total_count, probs=None, logits=None, validate_args=None):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        new = self._get_checked_instance(NegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(NegativeBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self._param.new(*args, **kwargs)

    @property
    def mean(self):
        if False:
            print('Hello World!')
        return self.total_count * torch.exp(self.logits)

    @property
    def mode(self):
        if False:
            i = 10
            return i + 15
        return ((self.total_count - 1) * self.logits.exp()).floor().clamp(min=0.0)

    @property
    def variance(self):
        if False:
            while True:
                i = 10
        return self.mean / torch.sigmoid(-self.logits)

    @lazy_property
    def logits(self):
        if False:
            i = 10
            return i + 15
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        if False:
            print('Hello World!')
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        if False:
            print('Hello World!')
        return self._param.size()

    @lazy_property
    def _gamma(self):
        if False:
            i = 10
            return i + 15
        return torch.distributions.Gamma(concentration=self.total_count, rate=torch.exp(-self.logits), validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        if False:
            while True:
                i = 10
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return torch.poisson(rate)

    def log_prob(self, value):
        if False:
            i = 10
            return i + 15
        if self._validate_args:
            self._validate_sample(value)
        log_unnormalized_prob = self.total_count * F.logsigmoid(-self.logits) + value * F.logsigmoid(self.logits)
        log_normalization = -torch.lgamma(self.total_count + value) + torch.lgamma(1.0 + value) + torch.lgamma(self.total_count)
        log_normalization = log_normalization.masked_fill(self.total_count + value == 0.0, 0.0)
        return log_unnormalized_prob - log_normalization