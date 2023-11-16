"""Implement various linear algebra algorithms for low rank matrices.
"""
__all__ = ['svd_lowrank', 'pca_lowrank']
from typing import Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function

def get_approximate_basis(A: Tensor, q: int, niter: Optional[int]=2, M: Optional[Tensor]=None) -> Tensor:
    if False:
        return 10
    "Return tensor :math:`Q` with :math:`q` orthonormal columns such\n    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is\n    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`\n    approximates :math:`A - M`.\n\n    .. note:: The implementation is based on the Algorithm 4.4 from\n              Halko et al, 2009.\n\n    .. note:: For an adequate approximation of a k-rank matrix\n              :math:`A`, where k is not known in advance but could be\n              estimated, the number of :math:`Q` columns, q, can be\n              choosen according to the following criteria: in general,\n              :math:`k <= q <= min(2*k, m, n)`. For large low-rank\n              matrices, take :math:`q = k + 5..10`.  If k is\n              relatively small compared to :math:`min(m, n)`, choosing\n              :math:`q = k + 0..2` may be sufficient.\n\n    .. note:: To obtain repeatable results, reset the seed for the\n              pseudorandom number generator\n\n    Args::\n        A (Tensor): the input tensor of size :math:`(*, m, n)`\n\n        q (int): the dimension of subspace spanned by :math:`Q`\n                 columns.\n\n        niter (int, optional): the number of subspace iterations to\n                               conduct; ``niter`` must be a\n                               nonnegative integer. In most cases, the\n                               default value 2 is more than enough.\n\n        M (Tensor, optional): the input tensor's mean of size\n                              :math:`(*, 1, n)`.\n\n    References::\n        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding\n          structure with randomness: probabilistic algorithms for\n          constructing approximate matrix decompositions,\n          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at\n          `arXiv <http://arxiv.org/abs/0909.4061>`_).\n    "
    niter = 2 if niter is None else niter
    (m, n) = A.shape[-2:]
    dtype = _utils.get_floating_dtype(A)
    matmul = _utils.matmul
    R = torch.randn(n, q, dtype=dtype, device=A.device)
    A_H = _utils.transjugate(A)
    if M is None:
        Q = torch.linalg.qr(matmul(A, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q)).Q
    else:
        M_H = _utils.transjugate(M)
        Q = torch.linalg.qr(matmul(A, R) - matmul(M, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q) - matmul(M_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q) - matmul(M, Q)).Q
    return Q

def svd_lowrank(A: Tensor, q: Optional[int]=6, niter: Optional[int]=2, M: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
    if False:
        for i in range(10):
            print('nop')
    "Return the singular value decomposition ``(U, S, V)`` of a matrix,\n    batches of matrices, or a sparse matrix :math:`A` such that\n    :math:`A \\approx U diag(S) V^T`. In case :math:`M` is given, then\n    SVD is computed for the matrix :math:`A - M`.\n\n    .. note:: The implementation is based on the Algorithm 5.1 from\n              Halko et al, 2009.\n\n    .. note:: To obtain repeatable results, reset the seed for the\n              pseudorandom number generator\n\n    .. note:: The input is assumed to be a low-rank matrix.\n\n    .. note:: In general, use the full-rank SVD implementation\n              :func:`torch.linalg.svd` for dense matrices due to its 10-fold\n              higher performance characteristics. The low-rank SVD\n              will be useful for huge sparse matrices that\n              :func:`torch.linalg.svd` cannot handle.\n\n    Args::\n        A (Tensor): the input tensor of size :math:`(*, m, n)`\n\n        q (int, optional): a slightly overestimated rank of A.\n\n        niter (int, optional): the number of subspace iterations to\n                               conduct; niter must be a nonnegative\n                               integer, and defaults to 2\n\n        M (Tensor, optional): the input tensor's mean of size\n                              :math:`(*, 1, n)`.\n\n    References::\n        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding\n          structure with randomness: probabilistic algorithms for\n          constructing approximate matrix decompositions,\n          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at\n          `arXiv <https://arxiv.org/abs/0909.4061>`_).\n\n    "
    if not torch.jit.is_scripting():
        tensor_ops = (A, M)
        if not set(map(type, tensor_ops)).issubset((torch.Tensor, type(None))) and has_torch_function(tensor_ops):
            return handle_torch_function(svd_lowrank, tensor_ops, A, q=q, niter=niter, M=M)
    return _svd_lowrank(A, q=q, niter=niter, M=M)

def _svd_lowrank(A: Tensor, q: Optional[int]=6, niter: Optional[int]=2, M: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
    if False:
        i = 10
        return i + 15
    q = 6 if q is None else q
    (m, n) = A.shape[-2:]
    matmul = _utils.matmul
    if M is None:
        M_t = None
    else:
        M_t = _utils.transpose(M)
    A_t = _utils.transpose(A)
    if m < n or n > q:
        Q = get_approximate_basis(A_t, q, niter=niter, M=M_t)
        Q_c = _utils.conjugate(Q)
        if M is None:
            B_t = matmul(A, Q_c)
        else:
            B_t = matmul(A, Q_c) - matmul(M, Q_c)
        assert B_t.shape[-2] == m, (B_t.shape, m)
        assert B_t.shape[-1] == q, (B_t.shape, q)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        (U, S, Vh) = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.mH
        V = Q.matmul(V)
    else:
        Q = get_approximate_basis(A, q, niter=niter, M=M)
        Q_c = _utils.conjugate(Q)
        if M is None:
            B = matmul(A_t, Q_c)
        else:
            B = matmul(A_t, Q_c) - matmul(M_t, Q_c)
        B_t = _utils.transpose(B)
        assert B_t.shape[-2] == q, (B_t.shape, q)
        assert B_t.shape[-1] == n, (B_t.shape, n)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        (U, S, Vh) = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.mH
        U = Q.matmul(U)
    return (U, S, V)

def pca_lowrank(A: Tensor, q: Optional[int]=None, center: bool=True, niter: int=2) -> Tuple[Tensor, Tensor, Tensor]:
    if False:
        i = 10
        return i + 15
    'Performs linear Principal Component Analysis (PCA) on a low-rank\n    matrix, batches of such matrices, or sparse matrix.\n\n    This function returns a namedtuple ``(U, S, V)`` which is the\n    nearly optimal approximation of a singular value decomposition of\n    a centered matrix :math:`A` such that :math:`A = U diag(S) V^T`.\n\n    .. note:: The relation of ``(U, S, V)`` to PCA is as follows:\n\n                - :math:`A` is a data matrix with ``m`` samples and\n                  ``n`` features\n\n                - the :math:`V` columns represent the principal directions\n\n                - :math:`S ** 2 / (m - 1)` contains the eigenvalues of\n                  :math:`A^T A / (m - 1)` which is the covariance of\n                  ``A`` when ``center=True`` is provided.\n\n                - ``matmul(A, V[:, :k])`` projects data to the first k\n                  principal components\n\n    .. note:: Different from the standard SVD, the size of returned\n              matrices depend on the specified rank and q\n              values as follows:\n\n                - :math:`U` is m x q matrix\n\n                - :math:`S` is q-vector\n\n                - :math:`V` is n x q matrix\n\n    .. note:: To obtain repeatable results, reset the seed for the\n              pseudorandom number generator\n\n    Args:\n\n        A (Tensor): the input tensor of size :math:`(*, m, n)`\n\n        q (int, optional): a slightly overestimated rank of\n                           :math:`A`. By default, ``q = min(6, m,\n                           n)``.\n\n        center (bool, optional): if True, center the input tensor,\n                                 otherwise, assume that the input is\n                                 centered.\n\n        niter (int, optional): the number of subspace iterations to\n                               conduct; niter must be a nonnegative\n                               integer, and defaults to 2.\n\n    References::\n\n        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding\n          structure with randomness: probabilistic algorithms for\n          constructing approximate matrix decompositions,\n          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at\n          `arXiv <http://arxiv.org/abs/0909.4061>`_).\n\n    '
    if not torch.jit.is_scripting():
        if type(A) is not torch.Tensor and has_torch_function((A,)):
            return handle_torch_function(pca_lowrank, (A,), A, q=q, center=center, niter=niter)
    (m, n) = A.shape[-2:]
    if q is None:
        q = min(6, m, n)
    elif not (q >= 0 and q <= min(m, n)):
        raise ValueError(f'q(={q}) must be non-negative integer and not greater than min(m, n)={min(m, n)}')
    if not niter >= 0:
        raise ValueError(f'niter(={niter}) must be non-negative integer')
    dtype = _utils.get_floating_dtype(A)
    if not center:
        return _svd_lowrank(A, q, niter=niter, M=None)
    if _utils.is_sparse(A):
        if len(A.shape) != 2:
            raise ValueError('pca_lowrank input is expected to be 2-dimensional tensor')
        c = torch.sparse.sum(A, dim=(-2,)) / m
        column_indices = c.indices()[0]
        indices = torch.zeros(2, len(column_indices), dtype=column_indices.dtype, device=column_indices.device)
        indices[0] = column_indices
        C_t = torch.sparse_coo_tensor(indices, c.values(), (n, 1), dtype=dtype, device=A.device)
        ones_m1_t = torch.ones(A.shape[:-2] + (1, m), dtype=dtype, device=A.device)
        M = _utils.transpose(torch.sparse.mm(C_t, ones_m1_t))
        return _svd_lowrank(A, q, niter=niter, M=M)
    else:
        C = A.mean(dim=(-2,), keepdim=True)
        return _svd_lowrank(A - C, q, niter=niter, M=None)