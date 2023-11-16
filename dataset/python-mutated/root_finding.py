"""Bisection implementation of alpha-entmax (Peters et al., 2019).

Backward pass wrt alpha per (Correia et al., 2019). See https://arxiv.org/pdf/1905.05702 for detailed description.
"""
import torch
import torch.nn as nn
from torch.autograd import Function

class EntmaxBisectFunction(Function):

    @classmethod
    def _gp(cls, x, alpha):
        if False:
            for i in range(10):
                print('nop')
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        if False:
            return 10
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        if False:
            i = 10
            return i + 15
        return cls._gp_inv(torch.clamp(X, min=0), alpha)

    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
        if False:
            return 10
        (p_m, backward_kwargs) = _entmax_bisect_forward(X, alpha, dim, n_iter, ensure_sum_one, cls)
        ctx.alpha = backward_kwargs['alpha']
        ctx.dim = backward_kwargs['dim']
        ctx.save_for_backward(p_m)
        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        if False:
            i = 10
            return i + 15
        (Y,) = ctx.saved_tensors
        gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        d_alpha = None
        if ctx.needs_input_grad[1]:
            S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
            ent = S.sum(ctx.dim).unsqueeze(ctx.dim)
            Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(ctx.dim)
            d_alpha = dY * (Y - Y_skewed) / (ctx.alpha - 1) ** 2
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(ctx.dim).unsqueeze(ctx.dim)
        return (dX, d_alpha, None, None, None)

def _entmax_bisect_forward(X, alpha, dim, n_iter, ensure_sum_one, cls=EntmaxBisectFunction):
    if False:
        print('Hello World!')
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)
    alpha_shape = list(X.shape)
    alpha_shape[dim] = 1
    alpha = alpha.expand(*alpha_shape)
    d = X.shape[dim]
    (max_val, _) = X.max(dim=dim, keepdim=True)
    X = X * (alpha - 1)
    max_val = max_val * (alpha - 1)
    tau_lo = max_val - cls._gp(1, alpha)
    tau_hi = max_val - cls._gp(1 / d, alpha)
    f_lo = cls._p(X - tau_lo, alpha).sum(dim) - 1
    dm = tau_hi - tau_lo
    for it in range(n_iter):
        dm /= 2
        tau_m = tau_lo + dm
        p_m = cls._p(X - tau_m, alpha)
        f_m = p_m.sum(dim) - 1
        mask = (f_m * f_lo >= 0).unsqueeze(dim)
        tau_lo = torch.where(mask, tau_m, tau_lo)
    if ensure_sum_one:
        p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)
    return (p_m, {'alpha': alpha, 'dim': dim})

class SparsemaxBisectFunction(EntmaxBisectFunction):

    @classmethod
    def _gp(cls, x, alpha):
        if False:
            return 10
        return x

    @classmethod
    def _gp_inv(cls, y, alpha):
        if False:
            print('Hello World!')
        return y

    @classmethod
    def _p(cls, x, alpha):
        if False:
            i = 10
            return i + 15
        return torch.clamp(x, min=0)

    @classmethod
    def forward(cls, ctx, X, dim=-1, n_iter=50, ensure_sum_one=True):
        if False:
            return 10
        (p_m, backward_kwargs) = _sparsemax_bisect_forward(X, dim, n_iter, ensure_sum_one)
        ctx.alpha = backward_kwargs['alpha']
        ctx.dim = backward_kwargs['dim']
        ctx.save_for_backward(p_m)
        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        if False:
            return 10
        (Y,) = ctx.saved_tensors
        gppr = (Y > 0).to(dtype=dY.dtype)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return (dX, None, None, None)

def _sparsemax_bisect_forward(X, dim, n_iter, ensure_sum_one):
    if False:
        for i in range(10):
            print('nop')
    return _entmax_bisect_forward(X, alpha=2, dim=dim, n_iter=50, ensure_sum_one=True, cls=SparsemaxBisectFunction)

def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True, training=True):
    if False:
        print('Hello World!')
    'alpha-entmax: normalizing sparse transform (a la softmax).\n\n    Solves the optimization problem:\n\n        max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.\n\n    where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,\n    using a bisection (root finding, binary search) algorithm.\n\n    This function is differentiable with respect to both X and alpha.\n\n    Parameters\n    ----------\n    X : torch.Tensor\n        The input tensor.\n\n    alpha : float or torch.Tensor\n        Tensor of alpha parameters (> 1) to use. If scalar\n        or python float, the same value is used for all rows, otherwise,\n        it must have shape (or be expandable to)\n        alpha.shape[j] == (X.shape[j] if j != dim else 1)\n        A value of alpha=2 corresponds to sparsemax, and alpha=1 would in theory recover\n        softmax. For numeric reasons, this algorithm does not work with `alpha=1`: if you\n        want softmax, we recommend `torch.nn.softmax`.\n\n    dim : int\n        The dimension along which to apply alpha-entmax.\n\n    n_iter : int\n        Number of bisection iterations. For float32, 24 iterations should\n        suffice for machine precision.\n\n    ensure_sum_one : bool,\n        Whether to divide the result by its sum. If false, the result might\n        sum to close but not exactly 1, which might cause downstream problems.\n\n    Returns\n    -------\n    P : torch tensor, same shape as X\n        The projection result, such that P.sum(dim=dim) == 1 elementwise.\n    '
    if not training:
        (output, _) = _entmax_bisect_forward(X, alpha, dim, n_iter, ensure_sum_one)
        return output
    return EntmaxBisectFunction.apply(X, alpha, dim, n_iter, ensure_sum_one)

def sparsemax_bisect(X, dim=-1, n_iter=50, ensure_sum_one=True, training=True):
    if False:
        return 10
    'sparsemax: normalizing sparse transform (a la softmax), via bisection.\n\n    Solves the projection:\n\n        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.\n\n    Parameters\n    ----------\n    X : torch.Tensor\n        The input tensor.\n\n    dim : int\n        The dimension along which to apply sparsemax.\n\n    n_iter : int\n        Number of bisection iterations. For float32, 24 iterations should\n        suffice for machine precision.\n\n    ensure_sum_one : bool,\n        Whether to divide the result by its sum. If false, the result might\n        sum to close but not exactly 1, which might cause downstream problems.\n\n    Note: This function does not yet support normalizing along anything except\n    the last dimension. Please use transposing and views to achieve more\n    general behavior.\n\n    Returns\n    -------\n    P : torch tensor, same shape as X\n        The projection result, such that P.sum(dim=dim) == 1 elementwise.\n    '
    if not training:
        (output, _) = _sparsemax_bisect_forward(X, dim, n_iter, ensure_sum_one)
        return output
    return SparsemaxBisectFunction.apply(X, dim, n_iter, ensure_sum_one)

class SparsemaxBisect(nn.Module):

    def __init__(self, dim=-1, n_iter=None):
        if False:
            while True:
                i = 10
        'sparsemax: normalizing sparse transform (a la softmax) via bisection\n\n        Solves the projection:\n\n            min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.\n\n        Parameters\n        ----------\n        dim : int\n            The dimension along which to apply sparsemax.\n\n        n_iter : int\n            Number of bisection iterations. For float32, 24 iterations should\n            suffice for machine precision.\n        '
        self.dim = dim
        self.n_iter = n_iter
        super().__init__()

    def forward(self, X):
        if False:
            while True:
                i = 10
        return sparsemax_bisect(X, dim=self.dim, n_iter=self.n_iter, training=self.training)

class EntmaxBisect(nn.Module):

    def __init__(self, alpha=1.5, dim=-1, n_iter=50):
        if False:
            i = 10
            return i + 15
        'alpha-entmax: normalizing sparse map (a la softmax) via bisection.\n\n        Solves the optimization problem:\n\n            max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.\n\n        where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,\n        using a bisection (root finding, binary search) algorithm.\n\n        Parameters\n        ----------\n        alpha : float or torch.Tensor\n            Tensor of alpha parameters (> 1) to use. If scalar\n            or python float, the same value is used for all rows, otherwise,\n            it must have shape (or be expandable to)\n            alpha.shape[j] == (X.shape[j] if j != dim else 1)\n            A value of alpha=2 corresponds to sparsemax; and alpha=1 would in theory recover\n            softmax. For numeric reasons, this algorithm does not work with `alpha=1`; if you\n            want softmax, we recommend `torch.nn.softmax`.\n\n        dim : int\n            The dimension along which to apply alpha-entmax.\n\n        n_iter : int\n            Number of bisection iterations. For float32, 24 iterations should\n            suffice for machine precision.\n\n        '
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        super().__init__()

    def forward(self, X):
        if False:
            i = 10
            return i + 15
        return entmax_bisect(X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter, training=self.training)