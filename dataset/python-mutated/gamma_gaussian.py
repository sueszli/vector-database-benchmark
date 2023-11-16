import math
import torch
from torch.distributions.utils import lazy_property
from torch.nn.functional import pad
from pyro.distributions.multivariate_studentt import MultivariateStudentT
from pyro.distributions.torch import MultivariateNormal
from pyro.distributions.util import broadcast_shape
from pyro.ops.tensor_utils import precision_to_scale_tril

class Gamma:
    """
    Non-normalized Gamma distribution.

        Gamma(concentration, rate) ~ (concentration - 1) * log(s) - rate * s
    """

    def __init__(self, log_normalizer, concentration, rate):
        if False:
            i = 10
            return i + 15
        self.log_normalizer = log_normalizer
        self.concentration = concentration
        self.rate = rate

    def log_density(self, s):
        if False:
            print('Hello World!')
        '\n        Non-normalized log probability of Gamma distribution.\n\n        This is mainly used for testing.\n        '
        return self.log_normalizer + (self.concentration - 1) * s.log() - self.rate * s

    def logsumexp(self):
        if False:
            i = 10
            return i + 15
        '\n        Integrates out the latent variable.\n        '
        return self.log_normalizer + torch.lgamma(self.concentration) - self.concentration * self.rate.log()

class GammaGaussian:
    """
    Non-normalized GammaGaussian distribution:

        GammaGaussian(x, s) ~ (concentration + 0.5 * dim - 1) * log(s)
                              - rate * s - s * 0.5 * info_vec.T @ inv(precision) @ info_vec)
                              - s * 0.5 * x.T @ precision @ x + s * x.T @ info_vec,

    which will be reparameterized as

        GammaGaussian(x, s) =: alpha * log(s) + s * (-0.5 * x.T @ precision @ x + x.T @ info_vec - beta).

    The `s` variable plays the role of a mixing variable such that

        p(x | s) ~ Gaussian(s * info_vec, s * precision).

    Conditioned on `s`, this represents an arbitrary semidefinite quadratic function,
    which can be interpreted as a rank-deficient Gaussian distribution.
    The precision matrix may have zero eigenvalues, thus it may be impossible
    to work directly with the covariance matrix.

    :param torch.Tensor log_normalizer: a normalization constant, which is mainly used to keep
        track of normalization terms during contractions.
    :param torch.Tensor info_vec: information vector, which is a scaled version of the mean
        ``info_vec = precision @ mean``. We use this represention to make gaussian contraction
        fast and stable.
    :param torch.Tensor precision: precision matrix of this gaussian.
    :param torch.Tensor alpha: reparameterized shape parameter of the marginal Gamma distribution of
        `s`. The shape parameter Gamma.concentration is reparameterized by:

            alpha = Gamma.concentration + 0.5 * dim - 1

    :param torch.Tensor beta: reparameterized rate parameter of the marginal Gamma distribution of
        `s`. The rate parameter Gamma.rate is reparameterized by:

            beta = Gamma.rate + 0.5 * info_vec.T @ inv(precision) @ info_vec
    """

    def __init__(self, log_normalizer, info_vec, precision, alpha, beta):
        if False:
            return 10
        assert info_vec.dim() >= 1
        assert precision.dim() >= 2
        assert precision.shape[-2:] == info_vec.shape[-1:] * 2
        self.log_normalizer = log_normalizer
        self.info_vec = info_vec
        self.precision = precision
        self.alpha = alpha
        self.beta = beta

    def dim(self):
        if False:
            for i in range(10):
                print('nop')
        return self.info_vec.size(-1)

    @lazy_property
    def batch_shape(self):
        if False:
            while True:
                i = 10
        return broadcast_shape(self.log_normalizer.shape, self.info_vec.shape[:-1], self.precision.shape[:-2], self.alpha.shape, self.beta.shape)

    def expand(self, batch_shape):
        if False:
            print('Hello World!')
        n = self.dim()
        log_normalizer = self.log_normalizer.expand(batch_shape)
        info_vec = self.info_vec.expand(batch_shape + (n,))
        precision = self.precision.expand(batch_shape + (n, n))
        alpha = self.alpha.expand(batch_shape)
        beta = self.beta.expand(batch_shape)
        return GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)

    def reshape(self, batch_shape):
        if False:
            i = 10
            return i + 15
        n = self.dim()
        log_normalizer = self.log_normalizer.reshape(batch_shape)
        info_vec = self.info_vec.reshape(batch_shape + (n,))
        precision = self.precision.reshape(batch_shape + (n, n))
        alpha = self.alpha.reshape(batch_shape)
        beta = self.beta.reshape(batch_shape)
        return GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        '\n        Index into the batch_shape of a GammaGaussian.\n        '
        assert isinstance(index, tuple)
        log_normalizer = self.log_normalizer[index]
        info_vec = self.info_vec[index + (slice(None),)]
        precision = self.precision[index + (slice(None), slice(None))]
        alpha = self.alpha[index]
        beta = self.beta[index]
        return GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)

    @staticmethod
    def cat(parts, dim=0):
        if False:
            print('Hello World!')
        '\n        Concatenate a list of GammaGaussians along a given batch dimension.\n        '
        if dim < 0:
            dim += len(parts[0].batch_shape)
        args = [torch.cat([getattr(g, attr) for g in parts], dim=dim) for attr in ['log_normalizer', 'info_vec', 'precision', 'alpha', 'beta']]
        return GammaGaussian(*args)

    def event_pad(self, left=0, right=0):
        if False:
            while True:
                i = 10
        '\n        Pad along event dimension.\n        '
        lr = (left, right)
        info_vec = pad(self.info_vec, lr)
        precision = pad(self.precision, lr + lr)
        return GammaGaussian(self.log_normalizer, info_vec, precision, self.alpha, self.beta)

    def event_permute(self, perm):
        if False:
            for i in range(10):
                print('nop')
        '\n        Permute along event dimension.\n        '
        assert isinstance(perm, torch.Tensor)
        assert perm.shape == (self.dim(),)
        info_vec = self.info_vec[..., perm]
        precision = self.precision[..., perm][..., perm, :]
        return GammaGaussian(self.log_normalizer, info_vec, precision, self.alpha, self.beta)

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds two GammaGaussians in log-density space.\n        '
        assert isinstance(other, GammaGaussian)
        assert self.dim() == other.dim()
        return GammaGaussian(self.log_normalizer + other.log_normalizer, self.info_vec + other.info_vec, self.precision + other.precision, self.alpha + other.alpha, self.beta + other.beta)

    def log_density(self, value, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        Evaluate the log density of this GammaGaussian at a point value::\n\n            alpha * log(s) + s * (-0.5 * value.T @ precision @ value + value.T @ info_vec - beta) + log_normalizer\n\n        This is mainly used for testing.\n        '
        if value.size(-1) == 0:
            batch_shape = broadcast_shape(value.shape[:-1], s.shape, self.batch_shape)
            return self.alpha * s.log() - self.beta * s + self.log_normalizer.expand(batch_shape)
        result = -0.5 * self.precision.matmul(value.unsqueeze(-1)).squeeze(-1)
        result = result + self.info_vec
        result = (value * result).sum(-1)
        return self.alpha * s.log() + (result - self.beta) * s + self.log_normalizer

    def condition(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Condition the Gaussian component on a trailing subset of its state.\n        This should satisfy::\n\n            g.condition(y).dim() == g.dim() - y.size(-1)\n\n        Note that since this is a non-normalized Gaussian, we include the\n        density of ``y`` in the result. Thus :meth:`condition` is similar to a\n        ``functools.partial`` binding of arguments::\n\n            left = x[..., :n]\n            right = x[..., n:]\n            g.log_density(x, s) == g.condition(right).log_density(left, s)\n        '
        assert isinstance(value, torch.Tensor)
        assert value.size(-1) <= self.info_vec.size(-1)
        n = self.dim() - value.size(-1)
        info_a = self.info_vec[..., :n]
        info_b = self.info_vec[..., n:]
        P_aa = self.precision[..., :n, :n]
        P_ab = self.precision[..., :n, n:]
        P_bb = self.precision[..., n:, n:]
        b = value
        info_vec = info_a - P_ab.matmul(b.unsqueeze(-1)).squeeze(-1)
        precision = P_aa
        log_normalizer = self.log_normalizer
        alpha = self.alpha
        beta = self.beta + 0.5 * P_bb.matmul(b.unsqueeze(-1)).squeeze(-1).mul(b).sum(-1) - b.mul(info_b).sum(-1)
        return GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)

    def marginalize(self, left=0, right=0):
        if False:
            print('Hello World!')
        '\n        Marginalizing out variables on either side of the event dimension::\n\n            g.marginalize(left=n).event_logsumexp() = g.event_logsumexp()\n            g.marginalize(right=n).event_logsumexp() = g.event_logsumexp()\n\n        and for data ``x``:\n\n            g.condition(x).event_logsumexp().log_density(s)\n              = g.marginalize(left=g.dim() - x.size(-1)).log_density(x, s)\n        '
        if left == 0 and right == 0:
            return self
        if left > 0 and right > 0:
            raise NotImplementedError
        n = self.dim()
        n_b = left + right
        a = slice(left, n - right)
        b = slice(None, left) if left else slice(n - right, None)
        P_aa = self.precision[..., a, a]
        P_ba = self.precision[..., b, a]
        P_bb = self.precision[..., b, b]
        P_b = torch.linalg.cholesky(P_bb)
        P_a = torch.linalg.solve_triangular(P_b, P_ba, upper=False)
        P_at = P_a.transpose(-1, -2)
        precision = P_aa - P_at.matmul(P_a)
        info_a = self.info_vec[..., a]
        info_b = self.info_vec[..., b]
        b_tmp = torch.linalg.solve_triangular(P_b, info_b.unsqueeze(-1), upper=False)
        info_vec = info_a
        if n_b < n:
            info_vec = info_vec - P_at.matmul(b_tmp).squeeze(-1)
        alpha = self.alpha - 0.5 * n_b
        beta = self.beta - 0.5 * b_tmp.squeeze(-1).pow(2).sum(-1)
        log_normalizer = self.log_normalizer + 0.5 * n_b * math.log(2 * math.pi) - P_b.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)

    def compound(self):
        if False:
            return 10
        '\n        Integrates out the latent multiplier `s`. The result will be a\n        Student-T distribution.\n        '
        concentration = self.alpha - 0.5 * self.dim() + 1
        scale_tril = precision_to_scale_tril(self.precision)
        scale_tril_t_u = scale_tril.transpose(-1, -2).matmul(self.info_vec.unsqueeze(-1)).squeeze(-1)
        u_Pinv_u = scale_tril_t_u.pow(2).sum(-1)
        rate = self.beta - 0.5 * u_Pinv_u
        loc = scale_tril.matmul(scale_tril_t_u.unsqueeze(-1)).squeeze(-1)
        scale_tril = scale_tril * (rate / concentration).sqrt().unsqueeze(-1).unsqueeze(-1)
        return MultivariateStudentT(2 * concentration, loc, scale_tril)

    def event_logsumexp(self):
        if False:
            print('Hello World!')
        '\n        Integrates out all latent state (i.e. operating on event dimensions) of Gaussian component.\n        '
        n = self.dim()
        chol_P = torch.linalg.cholesky(self.precision)
        chol_P_u = torch.linalg.solve_triangular(chol_P, self.info_vec.unsqueeze(-1), upper=False).squeeze(-1)
        u_P_u = chol_P_u.pow(2).sum(-1)
        concentration = self.alpha - 0.5 * n + 1
        rate = self.beta - 0.5 * u_P_u
        log_normalizer_tmp = 0.5 * n * math.log(2 * math.pi) - chol_P.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return Gamma(self.log_normalizer + log_normalizer_tmp, concentration, rate)

def gamma_and_mvn_to_gamma_gaussian(gamma, mvn):
    if False:
        i = 10
        return i + 15
    '\n    Convert a pair of Gamma and Gaussian distributions to a GammaGaussian.\n\n        p(x | s) ~ Gaussian(s * info_vec, s * precision)\n        p(s) ~ Gamma(alpha, beta)\n        p(x, s) ~ GammaGaussian(info_vec, precision, alpha, beta)\n\n    :param ~pyro.distributions.Gamma gamma: the mixing distribution\n    :param ~pyro.distributions.MultivariateNormal mvn: the conditional distribution\n        when mixing is 1.\n    :return: A GammaGaussian object.\n    :rtype: ~pyro.ops.gaussian_gamma.GammaGaussian\n    '
    assert isinstance(gamma, torch.distributions.Gamma)
    assert isinstance(mvn, torch.distributions.MultivariateNormal)
    n = mvn.loc.size(-1)
    precision = mvn.precision_matrix
    info_vec = precision.matmul(mvn.loc.unsqueeze(-1)).squeeze(-1)
    alpha = gamma.concentration + (0.5 * n - 1)
    beta = gamma.rate + 0.5 * (info_vec * mvn.loc).sum(-1)
    gaussian_logsumexp = 0.5 * n * math.log(2 * math.pi) + mvn.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    log_normalizer = -Gamma(gaussian_logsumexp, gamma.concentration, gamma.rate).logsumexp()
    return GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)

def scale_mvn(mvn, s):
    if False:
        return 10
    '\n    Transforms a MVN distribution to another MVN distribution according to\n\n        scale(mvn(loc, precision), s) := mvn(loc, s * precision).\n    '
    assert isinstance(mvn, torch.distributions.MultivariateNormal)
    assert isinstance(s, torch.Tensor)
    batch_shape = broadcast_shape(s.shape, mvn.batch_shape)
    loc = mvn.loc.expand(batch_shape + (-1,))
    scale_tril = mvn.scale_tril / s.sqrt().unsqueeze(-1).unsqueeze(-1)
    return MultivariateNormal(loc, scale_tril=scale_tril)

def matrix_and_mvn_to_gamma_gaussian(matrix, mvn):
    if False:
        while True:
            i = 10
    '\n    Convert a noisy affine function to a GammaGaussian, where the noise precision\n    is scaled by an auxiliary variable `s`. The noisy affine function (conditioned\n    on `s`) is defined as::\n\n        y = x @ matrix + scale(mvn, s).sample()\n\n    :param ~torch.Tensor matrix: A matrix with rightmost shape ``(x_dim, y_dim)``.\n    :param ~pyro.distributions.MultivariateNormal mvn: A multivariate normal distribution.\n    :return: A GammaGaussian with broadcasted batch shape and ``.dim() == x_dim + y_dim``.\n    :rtype: ~pyro.ops.gaussian_gamma.GammaGaussian\n    '
    assert isinstance(mvn, torch.distributions.MultivariateNormal)
    assert isinstance(matrix, torch.Tensor)
    (x_dim, y_dim) = matrix.shape[-2:]
    assert mvn.event_shape == (y_dim,)
    batch_shape = broadcast_shape(matrix.shape[:-2], mvn.batch_shape)
    matrix = matrix.expand(batch_shape + (x_dim, y_dim))
    mvn = mvn.expand(batch_shape)
    P_yy = mvn.precision_matrix
    neg_P_xy = matrix.matmul(P_yy)
    P_xy = -neg_P_xy
    P_yx = P_xy.transpose(-1, -2)
    P_xx = neg_P_xy.matmul(matrix.transpose(-1, -2))
    precision = torch.cat([torch.cat([P_xx, P_xy], -1), torch.cat([P_yx, P_yy], -1)], -2)
    info_y = P_yy.matmul(mvn.loc.unsqueeze(-1)).squeeze(-1)
    info_x = -matrix.matmul(info_y.unsqueeze(-1)).squeeze(-1)
    info_vec = torch.cat([info_x, info_y], -1)
    log_normalizer = -0.5 * y_dim * math.log(2 * math.pi) - mvn.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    beta = 0.5 * (info_y * mvn.loc).sum(-1)
    alpha = beta.new_full(beta.shape, 0.5 * y_dim)
    result = GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)
    assert result.batch_shape == batch_shape
    assert result.dim() == x_dim + y_dim
    return result

def gamma_gaussian_tensordot(x, y, dims=0):
    if False:
        print('Hello World!')
    '\n    Computes the integral over two GammaGaussians:\n\n        `(x @ y)((a,c),s) = log(integral(exp(x((a,b),s) + y((b,c),s)), b))`,\n\n    where `x` is a gaussian over variables (a,b), `y` is a gaussian over variables\n    (b,c), (a,b,c) can each be sets of zero or more variables, and `dims` is the size of b.\n\n    :param x: a GammaGaussian instance\n    :param y: a GammaGaussian instance\n    :param dims: number of variables to contract\n    '
    assert isinstance(x, GammaGaussian)
    assert isinstance(y, GammaGaussian)
    na = x.dim() - dims
    nb = dims
    nc = y.dim() - dims
    assert na >= 0
    assert nb >= 0
    assert nc >= 0
    device = x.info_vec.device
    perm = torch.cat([torch.arange(na, device=device), torch.arange(x.dim(), x.dim() + nc, device=device), torch.arange(na, x.dim(), device=device)])
    return (x.event_pad(right=nc) + y.event_pad(left=na)).event_permute(perm).marginalize(right=nb)