"""
=============================
Jacobians, hessians, and more
=============================

Computing jacobians or hessians are useful in a number of non-traditional
deep learning models. It is difficult (or annoying) to compute these quantities
efficiently using a standard autodiff system like PyTorch Autograd; functorch
provides ways of computing various higher-order autodiff quantities efficiently.
"""
from functools import partial
import torch
import torch.nn.functional as F
torch.manual_seed(0)

def predict(weight, bias, x):
    if False:
        return 10
    return F.linear(x, weight, bias).tanh()
D = 16
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D)
xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)

def compute_jac(xp):
    if False:
        for i in range(10):
            print('nop')
    jacobian_rows = [torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0] for vec in unit_vectors]
    return torch.stack(jacobian_rows)
jacobian = compute_jac(xp)
from functorch import vjp, vmap
(_, vjp_fn) = vjp(partial(predict, weight, bias), x)
(ft_jacobian,) = vmap(vjp_fn)(unit_vectors)
assert torch.allclose(ft_jacobian, jacobian)
from functorch import jacrev
ft_jacobian = jacrev(predict, argnums=2)(weight, bias, x)
assert torch.allclose(ft_jacobian, jacobian)
from torch.utils.benchmark import Timer
without_vmap = Timer(stmt='compute_jac(xp)', globals=globals())
with_vmap = Timer(stmt='jacrev(predict, argnums=2)(weight, bias, x)', globals=globals())
print(without_vmap.timeit(500))
print(with_vmap.timeit(500))
(ft_jac_weight, ft_jac_bias) = jacrev(predict, argnums=(0, 1))(weight, bias, x)
from functorch import jacfwd, jacrev
Din = 32
Dout = 2048
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)
using_fwd = Timer(stmt='jacfwd(predict, argnums=2)(weight, bias, x)', globals=globals())
using_bwd = Timer(stmt='jacrev(predict, argnums=2)(weight, bias, x)', globals=globals())
print(f'jacfwd time: {using_fwd.timeit(500)}')
print(f'jacrev time: {using_bwd.timeit(500)}')
Din = 2048
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)
using_fwd = Timer(stmt='jacfwd(predict, argnums=2)(weight, bias, x)', globals=globals())
using_bwd = Timer(stmt='jacrev(predict, argnums=2)(weight, bias, x)', globals=globals())
print(f'jacfwd time: {using_fwd.timeit(500)}')
print(f'jacrev time: {using_bwd.timeit(500)}')
from functorch import hessian
hess2 = jacrev(jacrev(predict, argnums=2), argnums=2)(weight, bias, x)

def predict_with_output_summed(weight, bias, x):
    if False:
        i = 10
        return i + 15
    return predict(weight, bias, x).sum(0)
batch_size = 64
Din = 31
Dout = 33
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(batch_size, Din)
batch_jacobian0 = jacrev(predict_with_output_summed, argnums=2)(weight, bias, x)
compute_batch_jacobian = vmap(jacrev(predict, argnums=2), in_dims=(None, None, 0))
batch_jacobian1 = compute_batch_jacobian(weight, bias, x)
assert torch.allclose(batch_jacobian0, batch_jacobian1)
compute_batch_hessian = vmap(hessian(predict, argnums=2), in_dims=(None, None, 0))
batch_hess = compute_batch_hessian(weight, bias, x)