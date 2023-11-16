import torch
from torch.distributions.utils import broadcast_all
import pyro.distributions.constraints as constraints
from pyro.distributions.rejector import Rejector
from pyro.distributions.torch import Exponential
from pyro.distributions.util import copy_docs_from, weakmethod

@copy_docs_from(Exponential)
class RejectionExponential(Rejector):
    arg_constraints = {'rate': constraints.positive, 'factor': constraints.positive}
    support = constraints.positive

    def __init__(self, rate, factor):
        if False:
            return 10
        assert (factor <= 1).all()
        (self.rate, self.factor) = broadcast_all(rate, factor)
        propose = Exponential(self.factor * self.rate)
        log_scale = self.factor.log()
        super().__init__(propose, self.log_prob_accept, log_scale)

    @weakmethod
    def log_prob_accept(self, x):
        if False:
            i = 10
            return i + 15
        result = (self.factor - 1) * self.rate * x
        assert result.max() <= 0, result.max()
        return result

    @property
    def batch_shape(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rate.shape

    @property
    def event_shape(self):
        if False:
            for i in range(10):
                print('nop')
        return torch.Size()