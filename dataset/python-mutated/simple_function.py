import time
import torch
from functorch import grad, make_fx
from functorch.compile import nnc_jit

def f(x):
    if False:
        return 10
    return torch.sin(x).sum()
inp = torch.randn(100)
grad_pt = grad(f)
grad_fx = make_fx(grad_pt)(inp)
grad_nnc = nnc_jit(grad_pt)

def bench(name, f, iters=10000, warmup=3):
    if False:
        print('Hello World!')
    for _ in range(warmup):
        f()
    begin = time.time()
    for _ in range(iters):
        f()
    print(f'{name}: ', time.time() - begin)
bench('Pytorch: ', lambda : grad_pt(inp))
bench('FX: ', lambda : grad_fx(inp))
bench('NNC: ', lambda : grad_nnc(inp))