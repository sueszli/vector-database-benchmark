import time
import torch
import torch.utils
from functorch.compile import aot_function, tvm_compile
a = torch.randn(2000, 1, 4, requires_grad=True)
b = torch.randn(1, 2000, 4)

def f(a):
    if False:
        while True:
            i = 10
    return (a * b).sum(dim=0)
fw_compiler = tvm_compile(target='llvm', tuning_logfile='fw_keops')
bw_compiler = tvm_compile(target='llvm', tuning_logfile='bw_keops')
compiled_f = aot_function(f, fw_compiler, bw_compiler)
iters = 10
out = compiled_f(a)
out.sum().backward()

def bench(func):
    if False:
        return 10
    begin = time.time()
    for _ in range(iters):
        out = func(a).sin()
        out.sum().backward()
        a.grad = None
    print(time.time() - begin)

def bench_jax():
    if False:
        for i in range(10):
            print('nop')
    import jax
    import jax.numpy as jnp
    jax_a = jnp.array(a.detach().numpy())
    jax_b = jnp.array(b.detach().numpy())

    def f(a):
        if False:
            for i in range(10):
                print('nop')
        return jnp.sin((a * jax_b).sum(axis=[0])).sum()
    jit_f = jax.jit(jax.grad(f))
    jit_f(jax_a)
    begin = time.time()
    for _ in range(iters):
        out = jit_f(jax_a)
    out.block_until_ready()
    print(time.time() - begin)
bench(f)
bench(compiled_f)