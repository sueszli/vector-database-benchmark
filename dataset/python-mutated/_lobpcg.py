"""Locally Optimal Block Preconditioned Conjugate Gradient methods.
"""
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
__all__ = ['lobpcg']

def _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U):
    if False:
        return 10
    F = D.unsqueeze(-2) - D.unsqueeze(-1)
    F.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    F.pow_(-1)
    Ut = U.mT.contiguous()
    res = torch.matmul(U, torch.matmul(torch.diag_embed(D_grad) + torch.matmul(Ut, U_grad) * F, Ut))
    return res

def _polynomial_coefficients_given_roots(roots):
    if False:
        print('Hello World!')
    "\n    Given the `roots` of a polynomial, find the polynomial's coefficients.\n\n    If roots = (r_1, ..., r_n), then the method returns\n    coefficients (a_0, a_1, ..., a_n (== 1)) so that\n    p(x) = (x - r_1) * ... * (x - r_n)\n         = x^n + a_{n-1} * x^{n-1} + ... a_1 * x_1 + a_0\n\n    Note: for better performance requires writing a low-level kernel\n    "
    poly_order = roots.shape[-1]
    poly_coeffs_shape = list(roots.shape)
    poly_coeffs_shape[-1] += 2
    poly_coeffs = roots.new_zeros(poly_coeffs_shape)
    poly_coeffs[..., 0] = 1
    poly_coeffs[..., -1] = 1
    for i in range(1, poly_order + 1):
        poly_coeffs_new = poly_coeffs.clone() if roots.requires_grad else poly_coeffs
        out = poly_coeffs_new.narrow(-1, poly_order - i, i + 1)
        out -= roots.narrow(-1, i - 1, 1) * poly_coeffs.narrow(-1, poly_order - i + 1, i + 1)
        poly_coeffs = poly_coeffs_new
    return poly_coeffs.narrow(-1, 1, poly_order + 1)

def _polynomial_value(poly, x, zero_power, transition):
    if False:
        i = 10
        return i + 15
    "\n    A generic method for computing poly(x) using the Horner's rule.\n\n    Args:\n      poly (Tensor): the (possibly batched) 1D Tensor representing\n                     polynomial coefficients such that\n                     poly[..., i] = (a_{i_0}, ..., a{i_n} (==1)), and\n                     poly(x) = poly[..., 0] * zero_power + ... + poly[..., n] * x^n\n\n      x (Tensor): the value (possible batched) to evalate the polynomial `poly` at.\n\n      zero_power (Tensor): the representation of `x^0`. It is application-specific.\n\n      transition (Callable): the function that accepts some intermediate result `int_val`,\n                             the `x` and a specific polynomial coefficient\n                             `poly[..., k]` for some iteration `k`.\n                             It basically performs one iteration of the Horner's rule\n                             defined as `x * int_val + poly[..., k] * zero_power`.\n                             Note that `zero_power` is not a parameter,\n                             because the step `+ poly[..., k] * zero_power` depends on `x`,\n                             whether it is a vector, a matrix, or something else, so this\n                             functionality is delegated to the user.\n    "
    res = zero_power.clone()
    for k in range(poly.size(-1) - 2, -1, -1):
        res = transition(res, x, poly[..., k])
    return res

def _matrix_polynomial_value(poly, x, zero_power=None):
    if False:
        while True:
            i = 10
    '\n    Evaluates `poly(x)` for the (batched) matrix input `x`.\n    Check out `_polynomial_value` function for more details.\n    '

    def transition(curr_poly_val, x, poly_coeff):
        if False:
            for i in range(10):
                print('nop')
        res = x.matmul(curr_poly_val)
        res.diagonal(dim1=-2, dim2=-1).add_(poly_coeff.unsqueeze(-1))
        return res
    if zero_power is None:
        zero_power = torch.eye(x.size(-1), x.size(-1), dtype=x.dtype, device=x.device).view(*[1] * len(list(x.shape[:-2])), x.size(-1), x.size(-1))
    return _polynomial_value(poly, x, zero_power, transition)

def _vector_polynomial_value(poly, x, zero_power=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Evaluates `poly(x)` for the (batched) vector input `x`.\n    Check out `_polynomial_value` function for more details.\n    '

    def transition(curr_poly_val, x, poly_coeff):
        if False:
            while True:
                i = 10
        res = torch.addcmul(poly_coeff.unsqueeze(-1), x, curr_poly_val)
        return res
    if zero_power is None:
        zero_power = x.new_ones(1).expand(x.shape)
    return _polynomial_value(poly, x, zero_power, transition)

def _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest):
    if False:
        return 10
    Ut = U.mT.contiguous()
    proj_U_ortho = -U.matmul(Ut)
    proj_U_ortho.diagonal(dim1=-2, dim2=-1).add_(1)
    gen = torch.Generator(A.device)
    U_ortho = proj_U_ortho.matmul(torch.randn((*A.shape[:-1], A.size(-1) - D.size(-1)), dtype=A.dtype, device=A.device, generator=gen))
    U_ortho_t = U_ortho.mT.contiguous()
    chr_poly_D = _polynomial_coefficients_given_roots(D)
    U_grad_projected = U_grad
    series_acc = U_grad_projected.new_zeros(U_grad_projected.shape)
    for k in range(1, chr_poly_D.size(-1)):
        poly_D = _vector_polynomial_value(chr_poly_D[..., k:], D)
        series_acc += U_grad_projected * poly_D.unsqueeze(-2)
        U_grad_projected = A.matmul(U_grad_projected)
    chr_poly_D_at_A = _matrix_polynomial_value(chr_poly_D, A)
    chr_poly_D_at_A_to_U_ortho = torch.matmul(U_ortho_t, torch.matmul(chr_poly_D_at_A, U_ortho))
    chr_poly_D_at_A_to_U_ortho_sign = -1 if largest and k % 2 == 1 else +1
    chr_poly_D_at_A_to_U_ortho_L = torch.linalg.cholesky(chr_poly_D_at_A_to_U_ortho_sign * chr_poly_D_at_A_to_U_ortho)
    res = _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)
    res -= U_ortho.matmul(chr_poly_D_at_A_to_U_ortho_sign * torch.cholesky_solve(U_ortho_t.matmul(series_acc), chr_poly_D_at_A_to_U_ortho_L)).matmul(Ut)
    return res

def _symeig_backward(D_grad, U_grad, A, D, U, largest):
    if False:
        print('Hello World!')
    if U.size(-1) == U.size(-2):
        return _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)
    else:
        return _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest)

class LOBPCGAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A: Tensor, k: Optional[int]=None, B: Optional[Tensor]=None, X: Optional[Tensor]=None, n: Optional[int]=None, iK: Optional[Tensor]=None, niter: Optional[int]=None, tol: Optional[float]=None, largest: Optional[bool]=None, method: Optional[str]=None, tracker: None=None, ortho_iparams: Optional[Dict[str, int]]=None, ortho_fparams: Optional[Dict[str, float]]=None, ortho_bparams: Optional[Dict[str, bool]]=None) -> Tuple[Tensor, Tensor]:
        if False:
            i = 10
            return i + 15
        A = A.contiguous() if not A.is_sparse else A
        if B is not None:
            B = B.contiguous() if not B.is_sparse else B
        (D, U) = _lobpcg(A, k, B, X, n, iK, niter, tol, largest, method, tracker, ortho_iparams, ortho_fparams, ortho_bparams)
        ctx.save_for_backward(A, B, D, U)
        ctx.largest = largest
        return (D, U)

    @staticmethod
    def backward(ctx, D_grad, U_grad):
        if False:
            for i in range(10):
                print('nop')
        A_grad = B_grad = None
        grads = [None] * 14
        (A, B, D, U) = ctx.saved_tensors
        largest = ctx.largest
        if A.is_sparse or (B is not None and B.is_sparse and ctx.needs_input_grad[2]):
            raise ValueError('lobpcg.backward does not support sparse input yet.Note that lobpcg.forward does though.')
        if A.dtype in (torch.complex64, torch.complex128) or (B is not None and B.dtype in (torch.complex64, torch.complex128)):
            raise ValueError('lobpcg.backward does not support complex input yet.Note that lobpcg.forward does though.')
        if B is not None:
            raise ValueError('lobpcg.backward does not support backward with B != I yet.')
        if largest is None:
            largest = True
        if B is None:
            A_grad = _symeig_backward(D_grad, U_grad, A, D, U, largest)
        grads[0] = A_grad
        grads[2] = B_grad
        return tuple(grads)

def lobpcg(A: Tensor, k: Optional[int]=None, B: Optional[Tensor]=None, X: Optional[Tensor]=None, n: Optional[int]=None, iK: Optional[Tensor]=None, niter: Optional[int]=None, tol: Optional[float]=None, largest: Optional[bool]=None, method: Optional[str]=None, tracker: None=None, ortho_iparams: Optional[Dict[str, int]]=None, ortho_fparams: Optional[Dict[str, float]]=None, ortho_bparams: Optional[Dict[str, bool]]=None) -> Tuple[Tensor, Tensor]:
    if False:
        while True:
            i = 10
    'Find the k largest (or smallest) eigenvalues and the corresponding\n    eigenvectors of a symmetric positive definite generalized\n    eigenvalue problem using matrix-free LOBPCG methods.\n\n    This function is a front-end to the following LOBPCG algorithms\n    selectable via `method` argument:\n\n      `method="basic"` - the LOBPCG method introduced by Andrew\n      Knyazev, see [Knyazev2001]. A less robust method, may fail when\n      Cholesky is applied to singular input.\n\n      `method="ortho"` - the LOBPCG method with orthogonal basis\n      selection [StathopoulosEtal2002]. A robust method.\n\n    Supported inputs are dense, sparse, and batches of dense matrices.\n\n    .. note:: In general, the basic method spends least time per\n      iteration. However, the robust methods converge much faster and\n      are more stable. So, the usage of the basic method is generally\n      not recommended but there exist cases where the usage of the\n      basic method may be preferred.\n\n    .. warning:: The backward method does not support sparse and complex inputs.\n      It works only when `B` is not provided (i.e. `B == None`).\n      We are actively working on extensions, and the details of\n      the algorithms are going to be published promptly.\n\n    .. warning:: While it is assumed that `A` is symmetric, `A.grad` is not.\n      To make sure that `A.grad` is symmetric, so that `A - t * A.grad` is symmetric\n      in first-order optimization routines, prior to running `lobpcg`\n      we do the following symmetrization map: `A -> (A + A.t()) / 2`.\n      The map is performed only when the `A` requires gradients.\n\n    Args:\n\n      A (Tensor): the input tensor of size :math:`(*, m, m)`\n\n      B (Tensor, optional): the input tensor of size :math:`(*, m,\n                  m)`. When not specified, `B` is interpreted as\n                  identity matrix.\n\n      X (tensor, optional): the input tensor of size :math:`(*, m, n)`\n                  where `k <= n <= m`. When specified, it is used as\n                  initial approximation of eigenvectors. X must be a\n                  dense tensor.\n\n      iK (tensor, optional): the input tensor of size :math:`(*, m,\n                  m)`. When specified, it will be used as preconditioner.\n\n      k (integer, optional): the number of requested\n                  eigenpairs. Default is the number of :math:`X`\n                  columns (when specified) or `1`.\n\n      n (integer, optional): if :math:`X` is not specified then `n`\n                  specifies the size of the generated random\n                  approximation of eigenvectors. Default value for `n`\n                  is `k`. If :math:`X` is specified, the value of `n`\n                  (when specified) must be the number of :math:`X`\n                  columns.\n\n      tol (float, optional): residual tolerance for stopping\n                 criterion. Default is `feps ** 0.5` where `feps` is\n                 smallest non-zero floating-point number of the given\n                 input tensor `A` data type.\n\n      largest (bool, optional): when True, solve the eigenproblem for\n                 the largest eigenvalues. Otherwise, solve the\n                 eigenproblem for smallest eigenvalues. Default is\n                 `True`.\n\n      method (str, optional): select LOBPCG method. See the\n                 description of the function above. Default is\n                 "ortho".\n\n      niter (int, optional): maximum number of iterations. When\n                 reached, the iteration process is hard-stopped and\n                 the current approximation of eigenpairs is returned.\n                 For infinite iteration but until convergence criteria\n                 is met, use `-1`.\n\n      tracker (callable, optional) : a function for tracing the\n                 iteration process. When specified, it is called at\n                 each iteration step with LOBPCG instance as an\n                 argument. The LOBPCG instance holds the full state of\n                 the iteration process in the following attributes:\n\n                   `iparams`, `fparams`, `bparams` - dictionaries of\n                   integer, float, and boolean valued input\n                   parameters, respectively\n\n                   `ivars`, `fvars`, `bvars`, `tvars` - dictionaries\n                   of integer, float, boolean, and Tensor valued\n                   iteration variables, respectively.\n\n                   `A`, `B`, `iK` - input Tensor arguments.\n\n                   `E`, `X`, `S`, `R` - iteration Tensor variables.\n\n                 For instance:\n\n                   `ivars["istep"]` - the current iteration step\n                   `X` - the current approximation of eigenvectors\n                   `E` - the current approximation of eigenvalues\n                   `R` - the current residual\n                   `ivars["converged_count"]` - the current number of converged eigenpairs\n                   `tvars["rerr"]` - the current state of convergence criteria\n\n                 Note that when `tracker` stores Tensor objects from\n                 the LOBPCG instance, it must make copies of these.\n\n                 If `tracker` sets `bvars["force_stop"] = True`, the\n                 iteration process will be hard-stopped.\n\n      ortho_iparams, ortho_fparams, ortho_bparams (dict, optional):\n                 various parameters to LOBPCG algorithm when using\n                 `method="ortho"`.\n\n    Returns:\n\n      E (Tensor): tensor of eigenvalues of size :math:`(*, k)`\n\n      X (Tensor): tensor of eigenvectors of size :math:`(*, m, k)`\n\n    References:\n\n      [Knyazev2001] Andrew V. Knyazev. (2001) Toward the Optimal\n      Preconditioned Eigensolver: Locally Optimal Block Preconditioned\n      Conjugate Gradient Method. SIAM J. Sci. Comput., 23(2),\n      517-541. (25 pages)\n      https://epubs.siam.org/doi/abs/10.1137/S1064827500366124\n\n      [StathopoulosEtal2002] Andreas Stathopoulos and Kesheng\n      Wu. (2002) A Block Orthogonalization Procedure with Constant\n      Synchronization Requirements. SIAM J. Sci. Comput., 23(6),\n      2165-2182. (18 pages)\n      https://epubs.siam.org/doi/10.1137/S1064827500370883\n\n      [DuerschEtal2018] Jed A. Duersch, Meiyue Shao, Chao Yang, Ming\n      Gu. (2018) A Robust and Efficient Implementation of LOBPCG.\n      SIAM J. Sci. Comput., 40(5), C655-C676. (22 pages)\n      https://epubs.siam.org/doi/abs/10.1137/17M1129830\n\n    '
    if not torch.jit.is_scripting():
        tensor_ops = (A, B, X, iK)
        if not set(map(type, tensor_ops)).issubset((torch.Tensor, type(None))) and has_torch_function(tensor_ops):
            return handle_torch_function(lobpcg, tensor_ops, A, k=k, B=B, X=X, n=n, iK=iK, niter=niter, tol=tol, largest=largest, method=method, tracker=tracker, ortho_iparams=ortho_iparams, ortho_fparams=ortho_fparams, ortho_bparams=ortho_bparams)
    if not torch._jit_internal.is_scripting():
        if A.requires_grad or (B is not None and B.requires_grad):
            A_sym = (A + A.mT) / 2
            B_sym = (B + B.mT) / 2 if B is not None else None
            return LOBPCGAutogradFunction.apply(A_sym, k, B_sym, X, n, iK, niter, tol, largest, method, tracker, ortho_iparams, ortho_fparams, ortho_bparams)
    elif A.requires_grad or (B is not None and B.requires_grad):
        raise RuntimeError('Script and require grads is not supported atm.If you just want to do the forward, use .detach()on A and B before calling into lobpcg')
    return _lobpcg(A, k, B, X, n, iK, niter, tol, largest, method, tracker, ortho_iparams, ortho_fparams, ortho_bparams)

def _lobpcg(A: Tensor, k: Optional[int]=None, B: Optional[Tensor]=None, X: Optional[Tensor]=None, n: Optional[int]=None, iK: Optional[Tensor]=None, niter: Optional[int]=None, tol: Optional[float]=None, largest: Optional[bool]=None, method: Optional[str]=None, tracker: None=None, ortho_iparams: Optional[Dict[str, int]]=None, ortho_fparams: Optional[Dict[str, float]]=None, ortho_bparams: Optional[Dict[str, bool]]=None) -> Tuple[Tensor, Tensor]:
    if False:
        i = 10
        return i + 15
    assert A.shape[-2] == A.shape[-1], A.shape
    if B is not None:
        assert A.shape == B.shape, (A.shape, B.shape)
    dtype = _utils.get_floating_dtype(A)
    device = A.device
    if tol is None:
        feps = {torch.float32: 1.2e-07, torch.float64: 2.23e-16}[dtype]
        tol = feps ** 0.5
    m = A.shape[-1]
    k = (1 if X is None else X.shape[-1]) if k is None else k
    n = (k if n is None else n) if X is None else X.shape[-1]
    if m < 3 * n:
        raise ValueError(f'LPBPCG algorithm is not applicable when the number of A rows (={m}) is smaller than 3 x the number of requested eigenpairs (={n})')
    method = 'ortho' if method is None else method
    iparams = {'m': m, 'n': n, 'k': k, 'niter': 1000 if niter is None else niter}
    fparams = {'tol': tol}
    bparams = {'largest': True if largest is None else largest}
    if method == 'ortho':
        if ortho_iparams is not None:
            iparams.update(ortho_iparams)
        if ortho_fparams is not None:
            fparams.update(ortho_fparams)
        if ortho_bparams is not None:
            bparams.update(ortho_bparams)
        iparams['ortho_i_max'] = iparams.get('ortho_i_max', 3)
        iparams['ortho_j_max'] = iparams.get('ortho_j_max', 3)
        fparams['ortho_tol'] = fparams.get('ortho_tol', tol)
        fparams['ortho_tol_drop'] = fparams.get('ortho_tol_drop', tol)
        fparams['ortho_tol_replace'] = fparams.get('ortho_tol_replace', tol)
        bparams['ortho_use_drop'] = bparams.get('ortho_use_drop', False)
    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker
    if len(A.shape) > 2:
        N = int(torch.prod(torch.tensor(A.shape[:-2])))
        bA = A.reshape((N,) + A.shape[-2:])
        bB = B.reshape((N,) + A.shape[-2:]) if B is not None else None
        bX = X.reshape((N,) + X.shape[-2:]) if X is not None else None
        bE = torch.empty((N, k), dtype=dtype, device=device)
        bXret = torch.empty((N, m, k), dtype=dtype, device=device)
        for i in range(N):
            A_ = bA[i]
            B_ = bB[i] if bB is not None else None
            X_ = torch.randn((m, n), dtype=dtype, device=device) if bX is None else bX[i]
            assert len(X_.shape) == 2 and X_.shape == (m, n), (X_.shape, (m, n))
            iparams['batch_index'] = i
            worker = LOBPCG(A_, B_, X_, iK, iparams, fparams, bparams, method, tracker)
            worker.run()
            bE[i] = worker.E[:k]
            bXret[i] = worker.X[:, :k]
        if not torch.jit.is_scripting():
            LOBPCG.call_tracker = LOBPCG_call_tracker_orig
        return (bE.reshape(A.shape[:-2] + (k,)), bXret.reshape(A.shape[:-2] + (m, k)))
    X = torch.randn((m, n), dtype=dtype, device=device) if X is None else X
    assert len(X.shape) == 2 and X.shape == (m, n), (X.shape, (m, n))
    worker = LOBPCG(A, B, X, iK, iparams, fparams, bparams, method, tracker)
    worker.run()
    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker_orig
    return (worker.E[:k], worker.X[:, :k])

class LOBPCG:
    """Worker class of LOBPCG methods."""

    def __init__(self, A: Optional[Tensor], B: Optional[Tensor], X: Tensor, iK: Optional[Tensor], iparams: Dict[str, int], fparams: Dict[str, float], bparams: Dict[str, bool], method: str, tracker: None) -> None:
        if False:
            print('Hello World!')
        self.A = A
        self.B = B
        self.iK = iK
        self.iparams = iparams
        self.fparams = fparams
        self.bparams = bparams
        self.method = method
        self.tracker = tracker
        m = iparams['m']
        n = iparams['n']
        self.X = X
        self.E = torch.zeros((n,), dtype=X.dtype, device=X.device)
        self.R = torch.zeros((m, n), dtype=X.dtype, device=X.device)
        self.S = torch.zeros((m, 3 * n), dtype=X.dtype, device=X.device)
        self.tvars: Dict[str, Tensor] = {}
        self.ivars: Dict[str, int] = {'istep': 0}
        self.fvars: Dict[str, float] = {'_': 0.0}
        self.bvars: Dict[str, bool] = {'_': False}

    def __str__(self):
        if False:
            i = 10
            return i + 15
        lines = ['LOPBCG:']
        lines += [f'  iparams={self.iparams}']
        lines += [f'  fparams={self.fparams}']
        lines += [f'  bparams={self.bparams}']
        lines += [f'  ivars={self.ivars}']
        lines += [f'  fvars={self.fvars}']
        lines += [f'  bvars={self.bvars}']
        lines += [f'  tvars={self.tvars}']
        lines += [f'  A={self.A}']
        lines += [f'  B={self.B}']
        lines += [f'  iK={self.iK}']
        lines += [f'  X={self.X}']
        lines += [f'  E={self.E}']
        r = ''
        for line in lines:
            r += line + '\n'
        return r

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        'Set and update iteration variables.'
        if self.ivars['istep'] == 0:
            X_norm = float(torch.norm(self.X))
            iX_norm = X_norm ** (-1)
            A_norm = float(torch.norm(_utils.matmul(self.A, self.X))) * iX_norm
            B_norm = float(torch.norm(_utils.matmul(self.B, self.X))) * iX_norm
            self.fvars['X_norm'] = X_norm
            self.fvars['A_norm'] = A_norm
            self.fvars['B_norm'] = B_norm
            self.ivars['iterations_left'] = self.iparams['niter']
            self.ivars['converged_count'] = 0
            self.ivars['converged_end'] = 0
        if self.method == 'ortho':
            self._update_ortho()
        else:
            self._update_basic()
        self.ivars['iterations_left'] = self.ivars['iterations_left'] - 1
        self.ivars['istep'] = self.ivars['istep'] + 1

    def update_residual(self):
        if False:
            for i in range(10):
                print('nop')
        'Update residual R from A, B, X, E.'
        mm = _utils.matmul
        self.R = mm(self.A, self.X) - mm(self.B, self.X) * self.E

    def update_converged_count(self):
        if False:
            i = 10
            return i + 15
        'Determine the number of converged eigenpairs using backward stable\n        convergence criterion, see discussion in Sec 4.3 of [DuerschEtal2018].\n\n        Users may redefine this method for custom convergence criteria.\n        '
        prev_count = self.ivars['converged_count']
        tol = self.fparams['tol']
        A_norm = self.fvars['A_norm']
        B_norm = self.fvars['B_norm']
        (E, X, R) = (self.E, self.X, self.R)
        rerr = torch.norm(R, 2, (0,)) * (torch.norm(X, 2, (0,)) * (A_norm + E[:X.shape[-1]] * B_norm)) ** (-1)
        converged = rerr < tol
        count = 0
        for b in converged:
            if not b:
                break
            count += 1
        assert count >= prev_count, f'the number of converged eigenpairs (was {prev_count}, got {count}) cannot decrease'
        self.ivars['converged_count'] = count
        self.tvars['rerr'] = rerr
        return count

    def stop_iteration(self):
        if False:
            i = 10
            return i + 15
        "Return True to stop iterations.\n\n        Note that tracker (if defined) can force-stop iterations by\n        setting ``worker.bvars['force_stop'] = True``.\n        "
        return self.bvars.get('force_stop', False) or self.ivars['iterations_left'] == 0 or self.ivars['converged_count'] >= self.iparams['k']

    def run(self):
        if False:
            return 10
        'Run LOBPCG iterations.\n\n        Use this method as a template for implementing LOBPCG\n        iteration scheme with custom tracker that is compatible with\n        TorchScript.\n        '
        self.update()
        if not torch.jit.is_scripting() and self.tracker is not None:
            self.call_tracker()
        while not self.stop_iteration():
            self.update()
            if not torch.jit.is_scripting() and self.tracker is not None:
                self.call_tracker()

    @torch.jit.unused
    def call_tracker(self):
        if False:
            print('Hello World!')
        'Interface for tracking iteration process in Python mode.\n\n        Tracking the iteration process is disabled in TorchScript\n        mode. In fact, one should specify tracker=None when JIT\n        compiling functions using lobpcg.\n        '
        pass

    def _update_basic(self):
        if False:
            while True:
                i = 10
        '\n        Update or initialize iteration variables when `method == "basic"`.\n        '
        mm = torch.matmul
        ns = self.ivars['converged_end']
        nc = self.ivars['converged_count']
        n = self.iparams['n']
        largest = self.bparams['largest']
        if self.ivars['istep'] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            (E, Z) = _utils.symeig(M, largest)
            self.X[:] = mm(self.X, mm(Ri, Z))
            self.E[:] = E
            np = 0
            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X
            W = _utils.matmul(self.iK, self.R)
            self.ivars['converged_end'] = ns = n + np + W.shape[-1]
            self.S[:, n + np:ns] = W
        else:
            S_ = self.S[:, nc:ns]
            Ri = self._get_rayleigh_ritz_transform(S_)
            M = _utils.qform(_utils.qform(self.A, S_), Ri)
            (E_, Z) = _utils.symeig(M, largest)
            self.X[:, nc:] = mm(S_, mm(Ri, Z[:, :n - nc]))
            self.E[nc:] = E_[:n - nc]
            P = mm(S_, mm(Ri, Z[:, n:2 * n - nc]))
            np = P.shape[-1]
            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X
            self.S[:, n:n + np] = P
            W = _utils.matmul(self.iK, self.R[:, nc:])
            self.ivars['converged_end'] = ns = n + np + W.shape[-1]
            self.S[:, n + np:ns] = W

    def _update_ortho(self):
        if False:
            while True:
                i = 10
        '\n        Update or initialize iteration variables when `method == "ortho"`.\n        '
        mm = torch.matmul
        ns = self.ivars['converged_end']
        nc = self.ivars['converged_count']
        n = self.iparams['n']
        largest = self.bparams['largest']
        if self.ivars['istep'] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            (E, Z) = _utils.symeig(M, largest)
            self.X = mm(self.X, mm(Ri, Z))
            self.update_residual()
            np = 0
            nc = self.update_converged_count()
            self.S[:, :n] = self.X
            W = self._get_ortho(self.R, self.X)
            ns = self.ivars['converged_end'] = n + np + W.shape[-1]
            self.S[:, n + np:ns] = W
        else:
            S_ = self.S[:, nc:ns]
            (E_, Z) = _utils.symeig(_utils.qform(self.A, S_), largest)
            self.X[:, nc:] = mm(S_, Z[:, :n - nc])
            self.E[nc:] = E_[:n - nc]
            P = mm(S_, mm(Z[:, n - nc:], _utils.basis(_utils.transpose(Z[:n - nc, n - nc:]))))
            np = P.shape[-1]
            self.update_residual()
            nc = self.update_converged_count()
            self.S[:, :n] = self.X
            self.S[:, n:n + np] = P
            W = self._get_ortho(self.R[:, nc:], self.S[:, :n + np])
            ns = self.ivars['converged_end'] = n + np + W.shape[-1]
            self.S[:, n + np:ns] = W

    def _get_rayleigh_ritz_transform(self, S):
        if False:
            for i in range(10):
                print('nop')
        'Return a transformation matrix that is used in Rayleigh-Ritz\n        procedure for reducing a general eigenvalue problem :math:`(S^TAS)\n        C = (S^TBS) C E` to a standard eigenvalue problem :math: `(Ri^T\n        S^TAS Ri) Z = Z E` where `C = Ri Z`.\n\n        .. note:: In the original Rayleight-Ritz procedure in\n          [DuerschEtal2018], the problem is formulated as follows::\n\n            SAS = S^T A S\n            SBS = S^T B S\n            D = (<diagonal matrix of SBS>) ** -1/2\n            R^T R = Cholesky(D SBS D)\n            Ri = D R^-1\n            solve symeig problem Ri^T SAS Ri Z = Theta Z\n            C = Ri Z\n\n          To reduce the number of matrix products (denoted by empty\n          space between matrices), here we introduce element-wise\n          products (denoted by symbol `*`) so that the Rayleight-Ritz\n          procedure becomes::\n\n            SAS = S^T A S\n            SBS = S^T B S\n            d = (<diagonal of SBS>) ** -1/2    # this is 1-d column vector\n            dd = d d^T                         # this is 2-d matrix\n            R^T R = Cholesky(dd * SBS)\n            Ri = R^-1 * d                      # broadcasting\n            solve symeig problem Ri^T SAS Ri Z = Theta Z\n            C = Ri Z\n\n          where `dd` is 2-d matrix that replaces matrix products `D M\n          D` with one element-wise product `M * dd`; and `d` replaces\n          matrix product `D M` with element-wise product `M *\n          d`. Also, creating the diagonal matrix `D` is avoided.\n\n        Args:\n        S (Tensor): the matrix basis for the search subspace, size is\n                    :math:`(m, n)`.\n\n        Returns:\n        Ri (tensor): upper-triangular transformation matrix of size\n                     :math:`(n, n)`.\n\n        '
        B = self.B
        mm = torch.matmul
        SBS = _utils.qform(B, S)
        d_row = SBS.diagonal(0, -2, -1) ** (-0.5)
        d_col = d_row.reshape(d_row.shape[0], 1)
        R = torch.linalg.cholesky(SBS * d_row * d_col, upper=True)
        return torch.linalg.solve_triangular(R, d_row.diag_embed(), upper=True, left=False)

    def _get_svqb(self, U: Tensor, drop: bool, tau: float) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Return B-orthonormal U.\n\n        .. note:: When `drop` is `False` then `svqb` is based on the\n                  Algorithm 4 from [DuerschPhD2015] that is a slight\n                  modification of the corresponding algorithm\n                  introduced in [StathopolousWu2002].\n\n        Args:\n\n          U (Tensor) : initial approximation, size is (m, n)\n          drop (bool) : when True, drop columns that\n                     contribution to the `span([U])` is small.\n          tau (float) : positive tolerance\n\n        Returns:\n\n          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`), size\n                       is (m, n1), where `n1 = n` if `drop` is `False,\n                       otherwise `n1 <= n`.\n\n        '
        if torch.numel(U) == 0:
            return U
        UBU = _utils.qform(self.B, U)
        d = UBU.diagonal(0, -2, -1)
        nz = torch.where(abs(d) != 0.0)
        assert len(nz) == 1, nz
        if len(nz[0]) < len(d):
            U = U[:, nz[0]]
            if torch.numel(U) == 0:
                return U
            UBU = _utils.qform(self.B, U)
            d = UBU.diagonal(0, -2, -1)
            nz = torch.where(abs(d) != 0.0)
            assert len(nz[0]) == len(d)
        d_col = (d ** (-0.5)).reshape(d.shape[0], 1)
        DUBUD = UBU * d_col * _utils.transpose(d_col)
        (E, Z) = _utils.symeig(DUBUD)
        t = tau * abs(E).max()
        if drop:
            keep = torch.where(E > t)
            assert len(keep) == 1, keep
            E = E[keep[0]]
            Z = Z[:, keep[0]]
            d_col = d_col[keep[0]]
        else:
            E[torch.where(E < t)[0]] = t
        return torch.matmul(U * _utils.transpose(d_col), Z * E ** (-0.5))

    def _get_ortho(self, U, V):
        if False:
            i = 10
            return i + 15
        'Return B-orthonormal U with columns are B-orthogonal to V.\n\n        .. note:: When `bparams["ortho_use_drop"] == False` then\n                  `_get_ortho` is based on the Algorithm 3 from\n                  [DuerschPhD2015] that is a slight modification of\n                  the corresponding algorithm introduced in\n                  [StathopolousWu2002]. Otherwise, the method\n                  implements Algorithm 6 from [DuerschPhD2015]\n\n        .. note:: If all U columns are B-collinear to V then the\n                  returned tensor U will be empty.\n\n        Args:\n\n          U (Tensor) : initial approximation, size is (m, n)\n          V (Tensor) : B-orthogonal external basis, size is (m, k)\n\n        Returns:\n\n          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`)\n                       such that :math:`V^T B U=0`, size is (m, n1),\n                       where `n1 = n` if `drop` is `False, otherwise\n                       `n1 <= n`.\n        '
        mm = torch.matmul
        mm_B = _utils.matmul
        m = self.iparams['m']
        tau_ortho = self.fparams['ortho_tol']
        tau_drop = self.fparams['ortho_tol_drop']
        tau_replace = self.fparams['ortho_tol_replace']
        i_max = self.iparams['ortho_i_max']
        j_max = self.iparams['ortho_j_max']
        use_drop = self.bparams['ortho_use_drop']
        for vkey in list(self.fvars.keys()):
            if vkey.startswith('ortho_') and vkey.endswith('_rerr'):
                self.fvars.pop(vkey)
        self.ivars.pop('ortho_i', 0)
        self.ivars.pop('ortho_j', 0)
        BV_norm = torch.norm(mm_B(self.B, V))
        BU = mm_B(self.B, U)
        VBU = mm(_utils.transpose(V), BU)
        i = j = 0
        stats = ''
        for i in range(i_max):
            U = U - mm(V, VBU)
            drop = False
            tau_svqb = tau_drop
            for j in range(j_max):
                if use_drop:
                    U = self._get_svqb(U, drop, tau_svqb)
                    drop = True
                    tau_svqb = tau_replace
                else:
                    U = self._get_svqb(U, False, tau_replace)
                if torch.numel(U) == 0:
                    self.ivars['ortho_i'] = i
                    self.ivars['ortho_j'] = j
                    return U
                BU = mm_B(self.B, U)
                UBU = mm(_utils.transpose(U), BU)
                U_norm = torch.norm(U)
                BU_norm = torch.norm(BU)
                R = UBU - torch.eye(UBU.shape[-1], device=UBU.device, dtype=UBU.dtype)
                R_norm = torch.norm(R)
                rerr = float(R_norm) * float(BU_norm * U_norm) ** (-1)
                vkey = f'ortho_UBUmI_rerr[{i}, {j}]'
                self.fvars[vkey] = rerr
                if rerr < tau_ortho:
                    break
            VBU = mm(_utils.transpose(V), BU)
            VBU_norm = torch.norm(VBU)
            U_norm = torch.norm(U)
            rerr = float(VBU_norm) * float(BV_norm * U_norm) ** (-1)
            vkey = f'ortho_VBU_rerr[{i}]'
            self.fvars[vkey] = rerr
            if rerr < tau_ortho:
                break
            if m < U.shape[-1] + V.shape[-1]:
                B = self.B
                assert B is not None
                raise ValueError(f'Overdetermined shape of U: #B-cols(={B.shape[-1]}) >= #U-cols(={U.shape[-1]}) + #V-cols(={V.shape[-1]}) must hold')
        self.ivars['ortho_i'] = i
        self.ivars['ortho_j'] = j
        return U
LOBPCG_call_tracker_orig = LOBPCG.call_tracker

def LOBPCG_call_tracker(self):
    if False:
        while True:
            i = 10
    self.tracker(self)