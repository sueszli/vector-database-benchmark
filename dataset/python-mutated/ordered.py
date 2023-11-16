import torch
from torch.distributions.transforms import Transform
from .. import constraints

class OrderedTransform(Transform):
    """
    Transforms a real vector into an ordered vector.

    Specifically, enforces monotonically increasing order on the last dimension
    of a given tensor via the transformation :math:`y_0 = x_0`,
    :math:`y_i = \\sum_{1 \\le j \\le i} \\exp(x_i)`
    """
    domain = constraints.real_vector
    codomain = constraints.ordered_vector
    bijective = True

    def _call(self, x):
        if False:
            for i in range(10):
                print('nop')
        z = torch.cat([x[..., :1], x[..., 1:].exp()], dim=-1)
        return torch.cumsum(z, dim=-1)

    def _inverse(self, y):
        if False:
            for i in range(10):
                print('nop')
        x = (y[..., 1:] - y[..., :-1]).log()
        return torch.cat([y[..., :1], x], dim=-1)

    def log_abs_det_jacobian(self, x, y):
        if False:
            print('Hello World!')
        return torch.sum(x[..., 1:], dim=-1)