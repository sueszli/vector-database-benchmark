import torch
from torch.distributions import constraints
from torch.nn import Parameter
import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.nn.module import PyroParam, pyro_method

class SparseGPRegression(GPModel):
    """
    Sparse Gaussian Process Regression model.

    In :class:`.GPRegression` model, when the number of input data :math:`X` is large,
    the covariance matrix :math:`k(X, X)` will require a lot of computational steps to
    compute its inverse (for log likelihood and for prediction). By introducing an
    additional inducing-input parameter :math:`X_u`, we can reduce computational cost
    by approximate :math:`k(X, X)` by a low-rank Nyström approximation :math:`Q`
    (see reference [1]), where

    .. math:: Q = k(X, X_u) k(X_u,X_u)^{-1} k(X_u, X).

    Given inputs :math:`X`, their noisy observations :math:`y`, and the inducing-input
    parameters :math:`X_u`, the model takes the form:

    .. math::
        u & \\sim \\mathcal{GP}(0, k(X_u, X_u)),\\\\
        f & \\sim q(f \\mid X, X_u) = \\mathbb{E}_{p(u)}q(f\\mid X, X_u, u),\\\\
        y & \\sim f + \\epsilon,

    where :math:`\\epsilon` is Gaussian noise and the conditional distribution
    :math:`q(f\\mid X, X_u, u)` is an approximation of

    .. math:: p(f\\mid X, X_u, u) = \\mathcal{N}(m, k(X, X) - Q),

    whose terms :math:`m` and :math:`k(X, X) - Q` is derived from the joint
    multivariate normal distribution:

    .. math:: [f, u] \\sim \\mathcal{GP}(0, k([X, X_u], [X, X_u])).

    This class implements three approximation methods:

    + Deterministic Training Conditional (DTC):

        .. math:: q(f\\mid X, X_u, u) = \\mathcal{N}(m, 0),

      which in turns will imply

        .. math:: f \\sim \\mathcal{N}(0, Q).

    + Fully Independent Training Conditional (FITC):

        .. math:: q(f\\mid X, X_u, u) = \\mathcal{N}(m, diag(k(X, X) - Q)),

      which in turns will correct the diagonal part of the approximation in DTC:

        .. math:: f \\sim \\mathcal{N}(0, Q + diag(k(X, X) - Q)).

    + Variational Free Energy (VFE), which is similar to DTC but has an additional
      `trace_term` in the model's log likelihood. This additional term makes "VFE"
      equivalent to the variational approach in :class:`.VariationalSparseGP`
      (see reference [2]).

    .. note:: This model has :math:`\\mathcal{O}(NM^2)` complexity for training,
        :math:`\\mathcal{O}(NM^2)` complexity for testing. Here, :math:`N` is the number
        of train inputs, :math:`M` is the number of inducing inputs.

    References:

    [1] `A Unifying View of Sparse Approximate Gaussian Process Regression`,
    Joaquin Quiñonero-Candela, Carl E. Rasmussen

    [2] `Variational learning of inducing variables in sparse Gaussian processes`,
    Michalis Titsias

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor Xu: Initial values for inducing points, which are parameters
        of our model.
    :param torch.Tensor noise: Variance of Gaussian noise of this model.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param str approx: One of approximation methods: "DTC", "FITC", and "VFE"
        (default).
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :param str name: Name of this model.
    """

    def __init__(self, X, y, kernel, Xu, noise=None, mean_function=None, approx=None, jitter=1e-06):
        if False:
            i = 10
            return i + 15
        assert isinstance(X, torch.Tensor), 'X needs to be a torch Tensor instead of a {}'.format(type(X))
        if y is not None:
            assert isinstance(y, torch.Tensor), 'y needs to be a torch Tensor instead of a {}'.format(type(y))
        assert isinstance(Xu, torch.Tensor), 'Xu needs to be a torch Tensor instead of a {}'.format(type(Xu))
        super().__init__(X, y, kernel, mean_function, jitter)
        self.Xu = Parameter(Xu)
        noise = self.X.new_tensor(1.0) if noise is None else noise
        self.noise = PyroParam(noise, constraints.positive)
        if approx is None:
            self.approx = 'VFE'
        elif approx in ['DTC', 'FITC', 'VFE']:
            self.approx = approx
        else:
            raise ValueError("The sparse approximation method should be one of 'DTC', 'FITC', 'VFE'.")

    @pyro_method
    def model(self):
        if False:
            while True:
                i = 10
        self.set_mode('model')
        N = self.X.size(0)
        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[::M + 1] += self.jitter
        Luu = torch.linalg.cholesky(Kuu)
        Kuf = self.kernel(self.Xu, self.X)
        W = torch.linalg.solve_triangular(Luu, Kuf, upper=False).t()
        D = self.noise.expand(N)
        if self.approx == 'FITC' or self.approx == 'VFE':
            Kffdiag = self.kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=-1)
            if self.approx == 'FITC':
                D = D + Kffdiag - Qffdiag
            else:
                trace_term = (Kffdiag - Qffdiag).sum() / self.noise
                trace_term = trace_term.clamp(min=0)
        zero_loc = self.X.new_zeros(N)
        f_loc = zero_loc + self.mean_function(self.X)
        if self.y is None:
            f_var = D + W.pow(2).sum(dim=-1)
            return (f_loc, f_var)
        else:
            if self.approx == 'VFE':
                pyro.factor(self._pyro_get_fullname('trace_term'), -trace_term / 2.0)
            return pyro.sample(self._pyro_get_fullname('y'), dist.LowRankMultivariateNormal(f_loc, W, D).expand_by(self.y.shape[:-1]).to_event(self.y.dim() - 1), obs=self.y)

    @pyro_method
    def guide(self):
        if False:
            while True:
                i = 10
        self.set_mode('guide')
        self._load_pyro_samples()

    def forward(self, Xnew, full_cov=False, noiseless=True):
        if False:
            return 10
        "\n        Computes the mean and covariance matrix (or variance) of Gaussian Process\n        posterior on a test input data :math:`X_{new}`:\n\n        .. math:: p(f^* \\mid X_{new}, X, y, k, X_u, \\epsilon) = \\mathcal{N}(loc, cov).\n\n        .. note:: The noise parameter ``noise`` (:math:`\\epsilon`), the inducing-point\n            parameter ``Xu``, together with kernel's parameters have been learned from\n            a training procedure (MCMC or SVI).\n\n        :param torch.Tensor Xnew: A input data for testing. Note that\n            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.\n        :param bool full_cov: A flag to decide if we want to predict full covariance\n            matrix or just variance.\n        :param bool noiseless: A flag to decide if we want to include noise in the\n            prediction output or not.\n        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`\n        :rtype: tuple(torch.Tensor, torch.Tensor)\n        "
        self._check_Xnew_shape(Xnew)
        self.set_mode('guide')
        N = self.X.size(0)
        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[::M + 1] += self.jitter
        Luu = torch.linalg.cholesky(Kuu)
        Kuf = self.kernel(self.Xu, self.X)
        W = torch.linalg.solve_triangular(Luu, Kuf, upper=False)
        D = self.noise.expand(N)
        if self.approx == 'FITC':
            Kffdiag = self.kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=0)
            D = D + Kffdiag - Qffdiag
        W_Dinv = W / D
        K = W_Dinv.matmul(W.t()).contiguous()
        K.view(-1)[::M + 1] += 1
        L = torch.linalg.cholesky(K)
        y_residual = self.y - self.mean_function(self.X)
        y_2D = y_residual.reshape(-1, N).t()
        W_Dinv_y = W_Dinv.matmul(y_2D)
        Kus = self.kernel(self.Xu, Xnew)
        Ws = torch.linalg.solve_triangular(Luu, Kus, upper=False)
        pack = torch.cat((W_Dinv_y, Ws), dim=1)
        Linv_pack = torch.linalg.solve_triangular(L, pack, upper=False)
        Linv_W_Dinv_y = Linv_pack[:, :W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1]:]
        C = Xnew.size(0)
        loc_shape = self.y.shape[:-1] + (C,)
        loc = Linv_W_Dinv_y.t().matmul(Linv_Ws).reshape(loc_shape)
        if full_cov:
            Kss = self.kernel(Xnew).contiguous()
            if not noiseless:
                Kss.view(-1)[::C + 1] += self.noise
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
            cov_shape = self.y.shape[:-1] + (C, C)
            cov = cov.expand(cov_shape)
        else:
            Kssdiag = self.kernel(Xnew, diag=True)
            if not noiseless:
                Kssdiag = Kssdiag + self.noise
            Qssdiag = Ws.pow(2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=0)
            cov_shape = self.y.shape[:-1] + (C,)
            cov = cov.expand(cov_shape)
        return (loc + self.mean_function(Xnew), cov)