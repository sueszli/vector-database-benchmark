import itertools
import numpy as np
import os
import pytest
import subprocess as subp
from neon import NervanaObject
from utils import call_func, gen_backend_tensors, tensors_allclose
try:
    from neon.backends.nervanagpu import NervanaGPU
except ImportError:

    class NervanaGPU(object):
        pass

class TestFuncs(object):
    """
    A collection of functions to be tested
    """

    @staticmethod
    def func_dot_reduction_mix(be, x0, x1, x2, x3, x4):
        if False:
            print('Hello World!')
        f1 = be.std(be.var(x0, axis=0, keepdims=True), axis=1, keepdims=True)
        f2 = be.max(x1, axis=0, keepdims=True) + be.min(x1, axis=0, keepdims=True)
        f3 = be.std(x2, keepdims=True)
        f4 = be.dot(1.0 / x3, x4 / x2)
        f5 = be.dot(x3, x4 - x0)
        f6 = be.dot(x2 / f4, f5 + x3)
        return f1 + f2 + f3 + f4 + 1.0 / be.dot(f5, f6)

    @staticmethod
    def func_dot_reduction_transpose_mix(be, x0, x1, x2, x3, x4):
        if False:
            while True:
                i = 10
        f1 = be.std(be.var(x0, axis=0, keepdims=True), axis=1, keepdims=True)
        f2 = be.max(x1, axis=0, keepdims=True) + be.min(x1, axis=0, keepdims=True)
        f3 = be.std(x2, keepdims=True).T
        f4 = be.dot(1.0 / x3, (x4 / x2).T).T
        f5 = be.dot(x3, (x4 - x0).T)
        f6 = be.dot(x2 / f4.T, f5 + x3).T
        return f1 + f2 + f3 + f4 + 1.0 / be.dot(f5, f6)

def pytest_generate_tests(metafunc):
    if False:
        i = 10
        return i + 15
    '\n    Test generator\n    '
    test_indices = [0]
    test_funcs = [TestFuncs.func_dot_reduction_mix, TestFuncs.func_dot_reduction_transpose_mix]
    test_tensor_flags = ['pos_rand', 'neg_rand', 'rand']
    test_tensor_dims = [(2, 2)]
    if 'custom_args' in metafunc.fixturenames:
        fargs = itertools.product(test_indices, test_funcs, test_tensor_flags, test_tensor_dims)
        metafunc.parametrize('custom_args', fargs)

def test_vs_numpy_mkl(backend_tests_mkl, custom_args):
    if False:
        for i in range(10):
            print('nop')
    (test_idx, f, flag, dim) = custom_args
    be = NervanaObject.be
    dtype = be.default_dtype
    tensors = gen_backend_tensors([np, be], [dim] * 5, [flag] * 5, dtype=dtype)
    numpy_func_val = call_func(f, np, tensors[0])
    backend_func_val = call_func(f, be, tensors[1])
    assert tensors_allclose(numpy_func_val, backend_func_val, rtol=0.01, atol=0.01)

@pytest.mark.hasgpu
def test_vs_numpy(backend_tests, custom_args):
    if False:
        for i in range(10):
            print('nop')
    (test_idx, f, flag, dim) = custom_args
    be = NervanaObject.be
    dtype = be.default_dtype
    tensors = gen_backend_tensors([np, be], [dim] * 5, [flag] * 5, dtype=dtype)
    numpy_func_val = call_func(f, np, tensors[0])
    backend_func_val = call_func(f, be, tensors[1])
    try:
        assert tensors_allclose(numpy_func_val, backend_func_val, rtol=0.01, atol=0.01)
    except AssertionError:
        if isinstance(NervanaObject.be, NervanaGPU):
            if os.getenv('PLATFORM'):
                platform = os.getenv('PLATFORM')
            elif os.path.exists('/usr/bin/nvidia-smi'):
                cmd = '/usr/bin/nvidia-smi -q | grep "Product Name" | tail -1 | cut -f 2 -d \':\' |                            cut -f 2,3 -d \' \''
                gpu_info = subp.check_output(cmd, shell=True)
            else:
                gpu_info = 'unknown'
            if gpu_info == 'TITAN Xp\n':
                platform = 'TITANXP'
            if platform == 'TITANXP':
                pytest.xfail(reason='xfail issue #854 with {} PLATFORM'.format(platform))
            else:
                assert tensors_allclose(numpy_func_val, backend_func_val, rtol=0.01, atol=0.01)