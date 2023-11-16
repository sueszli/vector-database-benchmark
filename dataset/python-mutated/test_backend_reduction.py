import itertools
import numpy as np
import pytest
from neon import NervanaObject
from utils import call_func, gen_backend_tensors, tensors_allclose

class TestFuncs(object):
    """
    A collection of functions to be tested
    """

    @staticmethod
    def func_reduction_mix_axis_0(be, x0, x1, x2, x3):
        if False:
            i = 10
            return i + 15
        f1 = be.mean(x0, axis=0, keepdims=True)
        f2 = be.max(x1, axis=0, keepdims=True) + be.min(x1, axis=0, keepdims=True)
        f3 = be.std(x2, keepdims=True)
        if be is np:
            f4 = be.argmax(x3, axis=0).reshape(-1, x3.shape[1])
        else:
            f4 = be.argmax(x3, axis=0, keepdims=True)
        x4 = be.empty((4, x0.shape[1]))
        x4[0, :] = f1
        x4[1, :] = f2
        x4[2, :] = f3
        x4[3, :] = f4
        return x4

    @staticmethod
    def func_reduction_mix_axis_1(be, x0, x1, x2, x3):
        if False:
            print('Hello World!')
        f1 = be.mean(x0, axis=1, keepdims=True)
        f2 = be.max(x1, axis=1, keepdims=True) + be.min(x1, axis=1, keepdims=True)
        f3 = be.std(x2, axis=1, keepdims=True)
        if be is np:
            f4 = be.argmax(x3, axis=1).reshape(x3.shape[0], -1)
        else:
            f4 = be.argmax(x3, axis=1, keepdims=True)
        if be is np:
            x4 = np.hstack([f1, f2, f3, f4])
        else:
            x4 = be.empty((x0.shape[0], 4))
            x4[:, 0] = f1
            x4[:, 1] = f2
            x4[:, 2] = f3
            x4[:, 3] = f4
        return x4

def pytest_generate_tests(metafunc):
    if False:
        i = 10
        return i + 15
    '\n    Test generator\n    '
    test_indices = [0]
    test_funcs = [TestFuncs.func_reduction_mix_axis_0, TestFuncs.func_reduction_mix_axis_1]
    test_tensor_flags = ['pos_rand', 'neg_rand', 'rand']
    test_tensor_dims = [(2, 2), (10, 32), (50, 50), (50, 128)]
    if 'custom_args' in metafunc.fixturenames:
        fargs = itertools.product(test_indices, test_funcs, test_tensor_flags, test_tensor_dims)
        metafunc.parametrize('custom_args', fargs)

@pytest.mark.hasgpu
def test_vs_numpy(backend_tests, custom_args):
    if False:
        return 10
    (test_idx, f, flag, dim) = custom_args
    be = NervanaObject.be
    dtype = be.default_dtype
    tensors = gen_backend_tensors([np, be], [dim] * 4, [flag] * 4, dtype=dtype)
    numpy_func_val = call_func(f, np, tensors[0])
    backend_func_val = call_func(f, be, tensors[1])
    assert tensors_allclose(numpy_func_val, backend_func_val, rtol=0.01, atol=0.01)

def test_vs_numpy_mkl(backend_tests_mkl, custom_args):
    if False:
        while True:
            i = 10
    (test_idx, f, flag, dim) = custom_args
    be = NervanaObject.be
    dtype = be.default_dtype
    tensors = gen_backend_tensors([np, be], [dim] * 4, [flag] * 4, dtype=dtype)
    numpy_func_val = call_func(f, np, tensors[0])
    backend_func_val = call_func(f, be, tensors[1])
    assert tensors_allclose(numpy_func_val, backend_func_val, rtol=0.01, atol=0.01)