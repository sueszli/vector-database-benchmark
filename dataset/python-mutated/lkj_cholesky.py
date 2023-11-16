"""
This closely follows the implementation in NumPyro (https://github.com/pyro-ppl/numpyro).

Original copyright notice:

# Copyright: Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
"""
import math
import torch
from torch.distributions import Beta, constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
__all__ = ['LKJCholesky']

class LKJCholesky(Distribution):
    """
    LKJ distribution for lower Cholesky factor of correlation matrices.
    The distribution is controlled by ``concentration`` parameter :math:`\\eta`
    to make the probability of the correlation matrix :math:`M` generated from
    a Cholesky factor proportional to :math:`\\det(M)^{\\eta - 1}`. Because of that,
    when ``concentration == 1``, we have a uniform distribution over Cholesky
    factors of correlation matrices::

        L ~ LKJCholesky(dim, concentration)
        X = L @ L' ~ LKJCorr(dim, concentration)

    Note that this distribution samples the
    Cholesky factor of correlation matrices and not the correlation matrices
    themselves and thereby differs slightly from the derivations in [1] for
    the `LKJCorr` distribution. For sampling, this uses the Onion method from
    [1] Section 3.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> l = LKJCholesky(3, 0.5)
        >>> l.sample()  # l @ l.T is a sample of a correlation 3x3 matrix
        tensor([[ 1.0000,  0.0000,  0.0000],
                [ 0.3516,  0.9361,  0.0000],
                [-0.1899,  0.4748,  0.8593]])

    Args:
        dimension (dim): dimension of the matrices
        concentration (float or Tensor): concentration/shape parameter of the
            distribution (often referred to as eta)

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method` (2009),
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    Journal of Multivariate Analysis. 100. 10.1016/j.jmva.2009.04.008
    """
    arg_constraints = {'concentration': constraints.positive}
    support = constraints.corr_cholesky

    def __init__(self, dim, concentration=1.0, validate_args=None):
        if False:
            for i in range(10):
                print('nop')
        if dim < 2:
            raise ValueError(f'Expected dim to be an integer greater than or equal to 2. Found dim={dim}.')
        self.dim = dim
        (self.concentration,) = broadcast_all(concentration)
        batch_shape = self.concentration.size()
        event_shape = torch.Size((dim, dim))
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = torch.arange(self.dim - 1, dtype=self.concentration.dtype, device=self.concentration.device)
        offset = torch.cat([offset.new_zeros((1,)), offset])
        beta_conc1 = offset + 0.5
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
        self._beta = Beta(beta_conc1, beta_conc0)
        super().__init__(batch_shape, event_shape, validate_args)

    def expand(self, batch_shape, _instance=None):
        if False:
            i = 10
            return i + 15
        new = self._get_checked_instance(LKJCholesky, _instance)
        batch_shape = torch.Size(batch_shape)
        new.dim = self.dim
        new.concentration = self.concentration.expand(batch_shape)
        new._beta = self._beta.expand(batch_shape + (self.dim,))
        super(LKJCholesky, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        if False:
            while True:
                i = 10
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        u_normal = torch.randn(self._extended_shape(sample_shape), dtype=y.dtype, device=y.device).tril(-1)
        u_hypersphere = u_normal / u_normal.norm(dim=-1, keepdim=True)
        u_hypersphere[..., 0, :].fill_(0.0)
        w = torch.sqrt(y) * u_hypersphere
        eps = torch.finfo(w.dtype).tiny
        diag_elems = torch.clamp(1 - torch.sum(w ** 2, dim=-1), min=eps).sqrt()
        w += torch.diag_embed(diag_elems)
        return w

    def log_prob(self, value):
        if False:
            while True:
                i = 10
        if self._validate_args:
            self._validate_sample(value)
        diag_elems = value.diagonal(dim1=-1, dim2=-2)[..., 1:]
        order = torch.arange(2, self.dim + 1, device=self.concentration.device)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order
        unnormalized_log_pdf = torch.sum(order * diag_elems.log(), dim=-1)
        dm1 = self.dim - 1
        alpha = self.concentration + 0.5 * dm1
        denominator = torch.lgamma(alpha) * dm1
        numerator = torch.mvlgamma(alpha - 0.5, dm1)
        pi_constant = 0.5 * dm1 * math.log(math.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized_log_pdf - normalize_term