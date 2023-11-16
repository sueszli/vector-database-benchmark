import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from pyro.distributions.constraints import unit_lower_cholesky

class UnitLowerCholeskyTransform(Transform):
    """
    Transform from unconstrained matrices to lower-triangular matrices with
    all ones diagonals.
    """
    domain = constraints.independent(constraints.real, 2)
    codomain = unit_lower_cholesky

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, UnitLowerCholeskyTransform)

    def _call(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x.tril(-1) + torch.eye(x.size(-1), device=x.device, dtype=x.dtype)

    def _inverse(self, y):
        if False:
            for i in range(10):
                print('nop')
        return y
__all__ = ['UnitLowerCholeskyTransform']