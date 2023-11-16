import inspect
import itertools
import sys
import time
import torch
from functorch import pointwise_operator
torch.set_num_threads(1)
torch._C._debug_set_fusion_group_inlining(False)

def rand(*shape):
    if False:
        while True:
            i = 10
    return torch.rand(*shape).mul(16).add(1)

def scalar():
    if False:
        for i in range(10):
            print('nop')
    return (rand(1), rand(1))

def small():
    if False:
        for i in range(10):
            print('nop')
    return (rand(32), rand(32))

def small_2d():
    if False:
        while True:
            i = 10
    return (rand(1, 32), rand(1, 32))

def small_broadcast():
    if False:
        i = 10
        return i + 15
    return (rand(4, 32), rand(32))

def medium():
    if False:
        return 10
    return (rand(32, 12, 64, 64), rand(32, 12, 64, 64))

def medium_sliced():
    if False:
        for i in range(10):
            print('nop')
    return (rand(32, 12, 64, 64)[..., ::2], rand(32, 12, 64, 64)[..., ::2])

def medium_transpose():
    if False:
        print('Hello World!')
    return (rand(32, 12, 64, 64).transpose(-1, -2), rand(32, 12, 64, 64).transpose(-1, -2))

def medium2():
    if False:
        i = 10
        return i + 15
    return (rand(32, 3, 224, 224), rand(32, 3, 224, 224))

def medium3d():
    if False:
        for i in range(10):
            print('nop')
    return (rand(16, 32, 64), rand(16, 32, 64))

def medium_channels_last():
    if False:
        return 10
    return (rand(32, 3, 224, 224).to(memory_format=torch.channels_last), rand(32, 3, 224, 224).to(memory_format=torch.channels_last))

def medium_broadcast():
    if False:
        i = 10
        return i + 15
    return (rand(32, 12, 64, 64), rand(64))

def medium_broadcast_channels_last():
    if False:
        i = 10
        return i + 15
    return (rand(32, 3, 223, 223).to(memory_format=torch.channels_last), rand(3, 1, 1))

def large():
    if False:
        while True:
            i = 10
    return (rand(8192, 8192), rand(8192, 8192))

def large_transpose():
    if False:
        i = 10
        return i + 15
    return (rand(8192, 8192).transpose(0, 1), rand(8192, 8192).transpose(0, 1))

def large_channels_last():
    if False:
        print('Hello World!')
    return (rand(32, 32, 256, 256).to(memory_format=torch.channels_last), rand(32, 32, 256, 256).to(memory_format=torch.channels_last))

def pathological_broadcast():
    if False:
        i = 10
        return i + 15
    return (rand(1, 32, 32, 2), rand(1024, 1, 1, 2))

def add(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a + b

def sub(a, b):
    if False:
        return 10
    return a - b

def mul(a, b):
    if False:
        while True:
            i = 10
    return a * b

def div(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a / b

def relu(a):
    if False:
        print('Hello World!')
    return a.relu()

def sigmoid(a):
    if False:
        i = 10
        return i + 15
    return a.sigmoid()

def tanh(a):
    if False:
        return 10
    return a.tanh()

def log(a):
    if False:
        for i in range(10):
            print('nop')
    return a.log()

def exp(a):
    if False:
        for i in range(10):
            print('nop')
    return a.exp()

def square(a):
    if False:
        print('Hello World!')
    return a ** 2

def fma(a, b):
    if False:
        while True:
            i = 10
    return a * b + b

def hardswish(a):
    if False:
        for i in range(10):
            print('nop')
    return a * (a + 3.0).clamp(0.0, 6.0) / 6.0

def native_hardswish(a):
    if False:
        print('Hello World!')
    return torch._C._nn.hardswish(a)

def softplus(a):
    if False:
        for i in range(10):
            print('nop')
    return (a * 1.0).exp().log1p() / 1.0

def mish(a):
    if False:
        i = 10
        return i + 15
    return a * ((a * 1.0).exp().log1p() / 1.0).tanh()

def time_cpu(fn, args, iters):
    if False:
        print('Hello World!')
    s = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    e = time.perf_counter()
    return e - s

def time_cuda(fn, args, iters):
    if False:
        i = 10
        return i + 15
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0

def benchmark_with_timer(fn, args, timer):
    if False:
        print('Hello World!')
    timer(fn, args, 3)
    calibration = timer(fn, args, 1)
    iters = int(1.0 / calibration)
    return timer(fn, args, iters) / iters

def benchmark(fn, args):
    if False:
        return 10
    timer = time_cpu if args[0].device.type == 'cpu' else time_cuda
    return benchmark_with_timer(fn, args, timer)

def micros(s):
    if False:
        return 10
    return f'{s * 1000000.0:.1f}'
shapes = [scalar, small, small_2d, small_broadcast, medium, medium2, medium3d, medium_sliced, medium_transpose, medium_channels_last, medium_broadcast, medium_broadcast_channels_last, large, large_transpose, large_channels_last, pathological_broadcast]
operators = [add, sub, mul, div, relu, sigmoid, tanh, log, exp, square, fma, hardswish, native_hardswish]
nope = set()
for (shape, operator) in itertools.product(shapes, operators):
    nargs = len(inspect.signature(operator).parameters)
    args = shape()[:nargs]
    try:
        if shape == medium_transpose:
            raise RuntimeError('pointwise_operator hangs on medium_transpose')
        pw_op = pointwise_operator(operator)
        torch.testing.assert_close(operator(*args), pw_op(*args))
    except Exception:
        print(f'pointwise_operator failed on {operator.__name__}, {shape.__name__}')
        nope.add((operator, shape))
    ts_op = torch.jit.script(operator)
    torch.testing.assert_close(operator(*args), ts_op(*args))
print('fuser,device,operator,shape,time')
results = []
for (shape, operator) in itertools.product(shapes, operators):
    nargs = len(inspect.signature(operator).parameters)
    args = shape()[:nargs]
    result = benchmark(operator, args)
    print(','.join(['eager', args[0].device.type, operator.__name__, shape.__name__, micros(result)]))
    try:
        if shape == medium_transpose:
            raise RuntimeError('pointwise_operator hangs on medium_transpose')
        if (operator, shape) in nope:
            raise RuntimeError('pointwise_operator fails on medium_transpose')
        pw_op = pointwise_operator(operator)
        result = benchmark(pw_op, args)
        print(','.join(['pointwise', args[0].device.type, operator.__name__, shape.__name__, micros(result)]))
    except Exception:
        print(','.join(['pointwise', args[0].device.type, operator.__name__, shape.__name__, micros(float('nan'))]))
    ts_op = torch.jit.script(operator)
    result = benchmark(ts_op, args)
    print(','.join(['fuser', args[0].device.type, operator.__name__, shape.__name__, micros(result)]))
    sys.stdout.flush()