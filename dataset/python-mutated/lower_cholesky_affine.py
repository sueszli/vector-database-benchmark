import torch
from torch.distributions.transforms import Transform
from .. import constraints
from ..util import copy_docs_from

@copy_docs_from(Transform)
class LowerCholeskyAffine(Transform):
    """
    A bijection of the form,

        :math:`\\mathbf{y} = \\mathbf{L} \\mathbf{x} + \\mathbf{r}`

    where `\\mathbf{L}` is a lower triangular matrix and `\\mathbf{r}` is a vector.

    :param loc: the fixed D-dimensional vector to shift the input by.
    :type loc: torch.tensor
    :param scale_tril: the D x D lower triangular matrix used in the transformation.
    :type scale_tril: torch.tensor

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    volume_preserving = False

    def __init__(self, loc, scale_tril, cache_size=0):
        if False:
            return 10
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale_tril = scale_tril
        assert loc.size(-1) == scale_tril.size(-1) == scale_tril.size(-2), 'loc and scale_tril must be of size D and D x D, respectively (instead: {}, {})'.format(loc.shape, scale_tril.shape)

    def _call(self, x):
        if False:
            return 10
        '\n        :param x: the input into the bijection\n        :type x: torch.Tensor\n\n        Invokes the bijection x=>y; in the prototypical context of a\n        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from\n        the base distribution (or the output of a previous transform)\n        '
        return torch.matmul(self.scale_tril, x.unsqueeze(-1)).squeeze(-1) + self.loc

    def _inverse(self, y):
        if False:
            i = 10
            return i + 15
        '\n        :param y: the output of the bijection\n        :type y: torch.Tensor\n\n        Inverts y => x.\n        '
        return torch.linalg.solve_triangular(self.scale_tril, (y - self.loc).unsqueeze(-1), upper=False).squeeze(-1)

    def log_abs_det_jacobian(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the elementwise determinant of the log Jacobian, i.e.\n        log(abs(dy/dx)).\n        '
        return torch.ones(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device) * self.scale_tril.diag().log().sum()

    def with_cache(self, cache_size=1):
        if False:
            while True:
                i = 10
        if self._cache_size == cache_size:
            return self
        return LowerCholeskyAffine(self.loc, self.scale_tril, cache_size=cache_size)