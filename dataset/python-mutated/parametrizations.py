from enum import Enum, auto
import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F
from typing import Optional
__all__ = ['orthogonal', 'spectral_norm', 'weight_norm']

def _is_orthogonal(Q, eps=None):
    if False:
        for i in range(10):
            print('nop')
    (n, k) = (Q.size(-2), Q.size(-1))
    Id = torch.eye(k, dtype=Q.dtype, device=Q.device)
    eps = 10.0 * n * torch.finfo(Q.dtype).eps
    return torch.allclose(Q.mH @ Q, Id, atol=eps)

def _make_orthogonal(A):
    if False:
        return 10
    'Assume that A is a tall matrix.\n\n    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative.\n    '
    (X, tau) = torch.geqrf(A)
    Q = torch.linalg.householder_product(X, tau)
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q

class _OrthMaps(Enum):
    matrix_exp = auto()
    cayley = auto()
    householder = auto()

class _Orthogonal(Module):
    base: Tensor

    def __init__(self, weight, orthogonal_map: _OrthMaps, *, use_trivialization=True) -> None:
        if False:
            return 10
        super().__init__()
        if weight.is_complex() and orthogonal_map == _OrthMaps.householder:
            raise ValueError('The householder parametrization does not support complex tensors.')
        self.shape = weight.shape
        self.orthogonal_map = orthogonal_map
        if use_trivialization:
            self.register_buffer('base', None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        (n, k) = (X.size(-2), X.size(-1))
        transposed = n < k
        if transposed:
            X = X.mT
            (n, k) = (k, n)
        if self.orthogonal_map == _OrthMaps.matrix_exp or self.orthogonal_map == _OrthMaps.cayley:
            X = X.tril()
            if n != k:
                X = torch.cat([X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
            A = X - X.mH
            if self.orthogonal_map == _OrthMaps.matrix_exp:
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == _OrthMaps.cayley:
                Id = torch.eye(n, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
            if n != k:
                Q = Q[..., :k]
        else:
            A = X.tril(diagonal=-1)
            tau = 2.0 / (1.0 + (A * A).sum(dim=-2))
            Q = torch.linalg.householder_product(A, tau)
            Q = Q * X.diagonal(dim1=-2, dim2=-1).int().unsqueeze(-2)
        if hasattr(self, 'base'):
            Q = self.base @ Q
        if transposed:
            Q = Q.mT
        return Q

    @torch.autograd.no_grad()
    def right_inverse(self, Q: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        if Q.shape != self.shape:
            raise ValueError(f'Expected a matrix or batch of matrices of shape {self.shape}. Got a tensor of shape {Q.shape}.')
        Q_init = Q
        (n, k) = (Q.size(-2), Q.size(-1))
        transpose = n < k
        if transpose:
            Q = Q.mT
            (n, k) = (k, n)
        if not hasattr(self, 'base'):
            if self.orthogonal_map == _OrthMaps.cayley or self.orthogonal_map == _OrthMaps.matrix_exp:
                raise NotImplementedError('It is not possible to assign to the matrix exponential or the Cayley parametrizations when use_trivialization=False.')
            (A, tau) = torch.geqrf(Q)
            A.diagonal(dim1=-2, dim2=-1).sign_()
            A.diagonal(dim1=-2, dim2=-1)[tau == 0.0] *= -1
            return A.mT if transpose else A
        else:
            if n == k:
                if not _is_orthogonal(Q):
                    Q = _make_orthogonal(Q)
                else:
                    Q = Q.clone()
            else:
                N = torch.randn(*Q.size()[:-2] + (n, n - k), dtype=Q.dtype, device=Q.device)
                Q = torch.cat([Q, N], dim=-1)
                Q = _make_orthogonal(Q)
            self.base = Q
            neg_Id = torch.zeros_like(Q_init)
            neg_Id.diagonal(dim1=-2, dim2=-1).fill_(-1.0)
            return neg_Id

def orthogonal(module: Module, name: str='weight', orthogonal_map: Optional[str]=None, *, use_trivialization: bool=True) -> Module:
    if False:
        while True:
            i = 10
    'Apply an orthogonal or unitary parametrization to a matrix or a batch of matrices.\n\n    Letting :math:`\\mathbb{K}` be :math:`\\mathbb{R}` or :math:`\\mathbb{C}`, the parametrized\n    matrix :math:`Q \\in \\mathbb{K}^{m \\times n}` is **orthogonal** as\n\n    .. math::\n\n        \\begin{align*}\n            Q^{\\text{H}}Q &= \\mathrm{I}_n \\mathrlap{\\qquad \\text{if }m \\geq n}\\\\\n            QQ^{\\text{H}} &= \\mathrm{I}_m \\mathrlap{\\qquad \\text{if }m < n}\n        \\end{align*}\n\n    where :math:`Q^{\\text{H}}` is the conjugate transpose when :math:`Q` is complex\n    and the transpose when :math:`Q` is real-valued, and\n    :math:`\\mathrm{I}_n` is the `n`-dimensional identity matrix.\n    In plain words, :math:`Q` will have orthonormal columns whenever :math:`m \\geq n`\n    and orthonormal rows otherwise.\n\n    If the tensor has more than two dimensions, we consider it as a batch of matrices of shape `(..., m, n)`.\n\n    The matrix :math:`Q` may be parametrized via three different ``orthogonal_map`` in terms of the original tensor:\n\n    - ``"matrix_exp"``/``"cayley"``:\n      the :func:`~torch.matrix_exp` :math:`Q = \\exp(A)` and the `Cayley map`_\n      :math:`Q = (\\mathrm{I}_n + A/2)(\\mathrm{I}_n - A/2)^{-1}` are applied to a skew-symmetric\n      :math:`A` to give an orthogonal matrix.\n    - ``"householder"``: computes a product of Householder reflectors\n      (:func:`~torch.linalg.householder_product`).\n\n    ``"matrix_exp"``/``"cayley"`` often make the parametrized weight converge faster than\n    ``"householder"``, but they are slower to compute for very thin or very wide matrices.\n\n    If ``use_trivialization=True`` (default), the parametrization implements the "Dynamic Trivialization Framework",\n    where an extra matrix :math:`B \\in \\mathbb{K}^{n \\times n}` is stored under\n    ``module.parametrizations.weight[0].base``. This helps the\n    convergence of the parametrized layer at the expense of some extra memory use.\n    See `Trivializations for Gradient-Based Optimization on Manifolds`_ .\n\n    Initial value of :math:`Q`:\n    If the original tensor is not parametrized and ``use_trivialization=True`` (default), the initial value\n    of :math:`Q` is that of the original tensor if it is orthogonal (or unitary in the complex case)\n    and it is orthogonalized via the QR decomposition otherwise (see :func:`torch.linalg.qr`).\n    Same happens when it is not parametrized and ``orthogonal_map="householder"`` even when ``use_trivialization=False``.\n    Otherwise, the initial value is the result of the composition of all the registered\n    parametrizations applied to the original tensor.\n\n    .. note::\n        This function is implemented using the parametrization functionality\n        in :func:`~torch.nn.utils.parametrize.register_parametrization`.\n\n\n    .. _`Cayley map`: https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map\n    .. _`Trivializations for Gradient-Based Optimization on Manifolds`: https://arxiv.org/abs/1909.09501\n\n    Args:\n        module (nn.Module): module on which to register the parametrization.\n        name (str, optional): name of the tensor to make orthogonal. Default: ``"weight"``.\n        orthogonal_map (str, optional): One of the following: ``"matrix_exp"``, ``"cayley"``, ``"householder"``.\n            Default: ``"matrix_exp"`` if the matrix is square or complex, ``"householder"`` otherwise.\n        use_trivialization (bool, optional): whether to use the dynamic trivialization framework.\n            Default: ``True``.\n\n    Returns:\n        The original module with an orthogonal parametrization registered to the specified\n        weight\n\n    Example::\n\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)\n        >>> orth_linear = orthogonal(nn.Linear(20, 40))\n        >>> orth_linear\n        ParametrizedLinear(\n        in_features=20, out_features=40, bias=True\n        (parametrizations): ModuleDict(\n            (weight): ParametrizationList(\n            (0): _Orthogonal()\n            )\n        )\n        )\n        >>> # xdoctest: +IGNORE_WANT\n        >>> Q = orth_linear.weight\n        >>> torch.dist(Q.T @ Q, torch.eye(20))\n        tensor(4.9332e-07)\n    '
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(f"Module '{module}' has no parameter or buffer with name '{name}'")
    if weight.ndim < 2:
        raise ValueError(f'Expected a matrix or batch of matrices. Got a tensor of {weight.ndim} dimensions.')
    if orthogonal_map is None:
        orthogonal_map = 'matrix_exp' if weight.size(-2) == weight.size(-1) or weight.is_complex() else 'householder'
    orth_enum = getattr(_OrthMaps, orthogonal_map, None)
    if orth_enum is None:
        raise ValueError(f'orthogonal_map has to be one of "matrix_exp", "cayley", "householder". Got: {orthogonal_map}')
    orth = _Orthogonal(weight, orth_enum, use_trivialization=use_trivialization)
    parametrize.register_parametrization(module, name, orth, unsafe=True)
    return module

class _WeightNorm(Module):

    def __init__(self, dim: Optional[int]=0) -> None:
        if False:
            return 10
        super().__init__()
        if dim is None:
            dim = -1
        self.dim = dim

    def forward(self, weight_g, weight_v):
        if False:
            while True:
                i = 10
        return torch._weight_norm(weight_v, weight_g, self.dim)

    def right_inverse(self, weight):
        if False:
            i = 10
            return i + 15
        weight_g = torch.norm_except_dim(weight, 2, self.dim)
        weight_v = weight
        return (weight_g, weight_v)

def weight_norm(module: Module, name: str='weight', dim: int=0):
    if False:
        while True:
            i = 10
    "Apply weight normalization to a parameter in the given module.\n\n    .. math::\n         \\mathbf{w} = g \\dfrac{\\mathbf{v}}{\\|\\mathbf{v}\\|}\n\n    Weight normalization is a reparameterization that decouples the magnitude\n    of a weight tensor from its direction. This replaces the parameter specified\n    by :attr:`name` with two parameters: one specifying the magnitude\n    and one specifying the direction.\n\n    By default, with ``dim=0``, the norm is computed independently per output\n    channel/plane. To compute a norm over the entire weight tensor, use\n    ``dim=None``.\n\n    See https://arxiv.org/abs/1602.07868\n\n    Args:\n        module (Module): containing module\n        name (str, optional): name of weight parameter\n        dim (int, optional): dimension over which to compute the norm\n\n    Returns:\n        The original module with the weight norm hook\n\n    Example::\n\n        >>> m = weight_norm(nn.Linear(20, 40), name='weight')\n        >>> m\n        ParametrizedLinear(\n          in_features=20, out_features=40, bias=True\n          (parametrizations): ModuleDict(\n            (weight): ParametrizationList(\n              (0): _WeightNorm()\n            )\n          )\n        )\n        >>> m.parametrizations.weight.original0.size()\n        torch.Size([40, 1])\n        >>> m.parametrizations.weight.original1.size()\n        torch.Size([40, 20])\n\n    "
    _weight_norm = _WeightNorm(dim)
    parametrize.register_parametrization(module, name, _weight_norm, unsafe=True)

    def _weight_norm_compat_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if False:
            print('Hello World!')
        g_key = f'{prefix}{name}_g'
        v_key = f'{prefix}{name}_v'
        if g_key in state_dict and v_key in state_dict:
            original0 = state_dict.pop(g_key)
            original1 = state_dict.pop(v_key)
            state_dict[f'{prefix}parametrizations.{name}.original0'] = original0
            state_dict[f'{prefix}parametrizations.{name}.original1'] = original1
    module._register_load_state_dict_pre_hook(_weight_norm_compat_hook)
    return module

class _SpectralNorm(Module):

    def __init__(self, weight: torch.Tensor, n_power_iterations: int=1, dim: int=0, eps: float=1e-12) -> None:
        if False:
            return 10
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError(f'Dimension out of range (expected to be in range of [-{ndim}, {ndim - 1}] but got {dim})')
        if n_power_iterations <= 0:
            raise ValueError(f'Expected n_power_iterations to be positive, but got n_power_iterations={n_power_iterations}')
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            (h, w) = weight_mat.size()
            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer('_u', F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer('_v', F.normalize(v, dim=0, eps=self.eps))
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        assert weight.ndim > 1
        if self.dim != 0:
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))
        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        if False:
            i = 10
            return i + 15
        assert weight_mat.ndim > 1
        for _ in range(n_power_iterations):
            self._u = F.normalize(torch.mv(weight_mat, self._v), dim=0, eps=self.eps, out=self._u)
            self._v = F.normalize(torch.mv(weight_mat.t(), self._u), dim=0, eps=self.eps, out=self._v)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        if weight.ndim == 1:
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            return weight / sigma

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        return value

def spectral_norm(module: Module, name: str='weight', n_power_iterations: int=1, eps: float=1e-12, dim: Optional[int]=None) -> Module:
    if False:
        for i in range(10):
            print('nop')
    'Apply spectral normalization to a parameter in the given module.\n\n    .. math::\n        \\mathbf{W}_{SN} = \\dfrac{\\mathbf{W}}{\\sigma(\\mathbf{W})},\n        \\sigma(\\mathbf{W}) = \\max_{\\mathbf{h}: \\mathbf{h} \\ne 0} \\dfrac{\\|\\mathbf{W} \\mathbf{h}\\|_2}{\\|\\mathbf{h}\\|_2}\n\n    When applied on a vector, it simplifies to\n\n    .. math::\n        \\mathbf{x}_{SN} = \\dfrac{\\mathbf{x}}{\\|\\mathbf{x}\\|_2}\n\n    Spectral normalization stabilizes the training of discriminators (critics)\n    in Generative Adversarial Networks (GANs) by reducing the Lipschitz constant\n    of the model. :math:`\\sigma` is approximated performing one iteration of the\n    `power method`_ every time the weight is accessed. If the dimension of the\n    weight tensor is greater than 2, it is reshaped to 2D in power iteration\n    method to get spectral norm.\n\n\n    See `Spectral Normalization for Generative Adversarial Networks`_ .\n\n    .. _`power method`: https://en.wikipedia.org/wiki/Power_iteration\n    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957\n\n    .. note::\n        This function is implemented using the parametrization functionality\n        in :func:`~torch.nn.utils.parametrize.register_parametrization`. It is a\n        reimplementation of :func:`torch.nn.utils.spectral_norm`.\n\n    .. note::\n        When this constraint is registered, the singular vectors associated to the largest\n        singular value are estimated rather than sampled at random. These are then updated\n        performing :attr:`n_power_iterations` of the `power method`_ whenever the tensor\n        is accessed with the module on `training` mode.\n\n    .. note::\n        If the `_SpectralNorm` module, i.e., `module.parametrization.weight[idx]`,\n        is in training mode on removal, it will perform another power iteration.\n        If you\'d like to avoid this iteration, set the module to eval mode\n        before its removal.\n\n    Args:\n        module (nn.Module): containing module\n        name (str, optional): name of weight parameter. Default: ``"weight"``.\n        n_power_iterations (int, optional): number of power iterations to\n            calculate spectral norm. Default: ``1``.\n        eps (float, optional): epsilon for numerical stability in\n            calculating norms. Default: ``1e-12``.\n        dim (int, optional): dimension corresponding to number of outputs.\n            Default: ``0``, except for modules that are instances of\n            ConvTranspose{1,2,3}d, when it is ``1``\n\n    Returns:\n        The original module with a new parametrization registered to the specified\n        weight\n\n    Example::\n\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)\n        >>> # xdoctest: +IGNORE_WANT("non-deterministic")\n        >>> snm = spectral_norm(nn.Linear(20, 40))\n        >>> snm\n        ParametrizedLinear(\n          in_features=20, out_features=40, bias=True\n          (parametrizations): ModuleDict(\n            (weight): ParametrizationList(\n              (0): _SpectralNorm()\n            )\n          )\n        )\n        >>> torch.linalg.matrix_norm(snm.weight, 2)\n        tensor(1.0081, grad_fn=<AmaxBackward0>)\n    '
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(f"Module '{module}' has no parameter or buffer with name '{name}'")
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(module, name, _SpectralNorm(weight, n_power_iterations, dim, eps))
    return module