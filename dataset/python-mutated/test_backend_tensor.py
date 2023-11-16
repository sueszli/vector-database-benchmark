"""
Test of basic math operations on the Tensors and compare with numpy results
The Tensor types includes GPU, MKL, and CPU Tensors
"""
from __future__ import print_function
import numpy as np
import itertools as itt
import pytest
from utils import tensors_allclose, allclose_with_out

def init_helper(lib, inA, inB, dtype):
    if False:
        while True:
            i = 10
    A = lib.array(inA, dtype=dtype)
    B = lib.array(inB, dtype=dtype)
    C = lib.empty(inB.shape, dtype=dtype)
    return (A, B, C)

def math_helper(lib, op, inA, inB, dtype):
    if False:
        return 10
    (A, B, C) = init_helper(lib, inA, inB, dtype)
    if op == '+':
        C[:] = A + B
    elif op == '-':
        C[:] = A - B
    elif op == '*':
        C[:] = A * B
    elif op == '/':
        C[:] = A / B
    elif op == '>':
        C[:] = A > B
    elif op == '>=':
        C[:] = A >= B
    elif op == '<':
        C[:] = A < B
    elif op == '<=':
        C[:] = A <= B
    return C

def init_helper_mkl(nm, inA, inB, dtype):
    if False:
        return 10
    A = nm.array(inA, dtype=dtype)
    B = nm.array(inB, dtype=dtype)
    C = nm.empty(inB.shape, dtype=dtype)
    return (A, B, C)

def math_helper_mkl(nm, op, inA, inB, dtype):
    if False:
        i = 10
        return i + 15
    (A, B, C) = init_helper_mkl(nm, inA, inB, dtype)
    if op == '+':
        C[:] = A + B
    elif op == '-':
        C[:] = A - B
    elif op == '*':
        C[:] = A * B
    elif op == '/':
        C[:] = A / B
    elif op == '>':
        C[:] = A > B
    elif op == '>=':
        C[:] = A >= B
    elif op == '<':
        C[:] = A < B
    elif op == '<=':
        C[:] = A <= B
    return C

def compare_helper(op, inA, inB, ng, nc, dtype):
    if False:
        for i in range(10):
            print('nop')
    numpy_result = math_helper(np, op, inA, inB, dtype=np.float32).astype(dtype)
    nervanaGPU_result = math_helper(ng, op, inA, inB, dtype=dtype).get()
    allclose_with_out(numpy_result, nervanaGPU_result, rtol=0, atol=1e-05)
    nervanaCPU_result = math_helper(nc, op, inA, inB, dtype=dtype).get()
    allclose_with_out(numpy_result, nervanaCPU_result, rtol=0, atol=1e-05)

def compare_helper_cpu(op, inA, inB, nc, dtype):
    if False:
        return 10
    numpy_result = math_helper(np, op, inA, inB, dtype=np.float32).astype(dtype)
    nervanaCPU_result = math_helper(nc, op, inA, inB, dtype=dtype).get()
    allclose_with_out(numpy_result, nervanaCPU_result, rtol=0, atol=1e-05)

def compare_helper_mkl(op, inA, inB, nm, dtype):
    if False:
        print('Hello World!')
    numpy_result = math_helper(np, op, inA, inB, dtype=np.float32).astype(dtype)
    nervanaMKL_result = math_helper_mkl(nm, op, inA, inB, dtype=dtype).get()
    allclose_with_out(numpy_result, nervanaMKL_result, rtol=0, atol=1e-05)

def rand_unif(dtype, dims):
    if False:
        i = 10
        return i + 15
    if np.dtype(dtype).kind == 'f':
        return np.random.uniform(-1, 1, dims).astype(dtype)
    else:
        iinfo = np.iinfo(dtype)
        return np.around(np.random.uniform(iinfo.min, iinfo.max, dims)).clip(iinfo.min, iinfo.max)

def pytest_generate_tests(metafunc):
    if False:
        i = 10
        return i + 15
    '\n    Build a list of test arguments.\n\n    '
    dims = [(64, 327), (64, 1), (1, 1023), (4, 3)]
    if 'fargs_tests' in metafunc.fixturenames:
        fargs = itt.product(dims)
        metafunc.parametrize('fargs_tests', fargs)

def test_slicing_mkl(fargs_tests, backend_pair_dtype_mkl_32):
    if False:
        for i in range(10):
            print('nop')
    dims = fargs_tests[0]
    (mkl, cpu) = backend_pair_dtype_mkl_32
    dtype = mkl.default_dtype
    array_np = np.random.uniform(-1, 1, dims).astype(dtype)
    array_nc = cpu.array(array_np, dtype=dtype)
    array_nm = mkl.array(array_np, dtype=dtype)
    assert tensors_allclose(array_nm[0], array_nc[0], rtol=0, atol=0.001)
    assert tensors_allclose(array_nm[-1], array_nc[-1], rtol=0, atol=0.001)
    assert tensors_allclose(array_nm[0, :], array_nc[0, :], rtol=0, atol=0.001)
    assert tensors_allclose(array_nm[0:], array_nc[0:], rtol=0, atol=0.001)
    assert tensors_allclose(array_nm[:-1], array_nc[:-1], rtol=0, atol=0.001)
    assert tensors_allclose(array_nm[:, 0], array_nc[:, 0], rtol=0, atol=0.001)
    assert tensors_allclose(array_nm[:, 0:1], array_nc[:, 0:1], rtol=0, atol=0.001)
    assert tensors_allclose(array_nm[-1, 0:], array_nc[-1:, 0:], rtol=0, atol=0.001)
    array_nc[0] = 0
    array_nm[0] = 0
    assert tensors_allclose(array_nm, array_nc, rtol=0, atol=0.001)

def test_reshape_separate_mkl(fargs_tests, backend_pair_dtype_mkl_32):
    if False:
        print('Hello World!')
    dims = fargs_tests[0]
    (mkl, cpu) = backend_pair_dtype_mkl_32
    dtype = mkl.default_dtype
    array_np = np.random.uniform(-1, 1, dims).astype(dtype)
    array_nc = cpu.array(array_np, dtype=dtype)
    array_nm = mkl.array(array_np, dtype=dtype)
    if dims[0] % 2 == 0:
        reshaped_nc = array_nc.reshape((2, dims[0] // 2, dims[1]))
        reshaped_nm = array_nm.reshape((2, dims[0] // 2, dims[1]))
        assert tensors_allclose(reshaped_nm, reshaped_nc, rtol=0, atol=1e-06)

def test_reshape_combine_mkl(fargs_tests, backend_pair_dtype_mkl_32):
    if False:
        for i in range(10):
            print('nop')
    dims = fargs_tests[0]
    (mkl, cpu) = backend_pair_dtype_mkl_32
    dtype = mkl.default_dtype
    if dims[0] % 2 == 0:
        orig_shape = (2, dims[0] // 2, dims[1])
        array_np = np.random.uniform(-1, 1, orig_shape).astype(dtype)
        array_nc = cpu.array(array_np, dtype=dtype)
        array_nm = mkl.array(array_np, dtype=dtype)
        reshaped_nc = array_nc.reshape(dims)
        reshaped_nm = array_nm.reshape(dims)
        assert tensors_allclose(reshaped_nm, reshaped_nc, rtol=0, atol=1e-06)

def test_math_mkl(fargs_tests, backend_mkl):
    if False:
        print('Hello World!')
    dims = fargs_tests[0]
    nm = backend_mkl
    dtype = nm.default_dtype
    randA = rand_unif(dtype, dims)
    randB = rand_unif(dtype, dims)
    compare_helper_mkl('+', randA, randB, nm, dtype)
    compare_helper_mkl('-', randA, randB, nm, dtype)
    compare_helper_mkl('*', randA, randB, nm, dtype)
    compare_helper_mkl('>', randA, randB, nm, dtype)
    compare_helper_mkl('>=', randA, randB, nm, dtype)
    compare_helper_mkl('<', randA, randB, nm, dtype)
    compare_helper_mkl('<=', randA, randB, nm, dtype)

def test_math_cpu(fargs_tests, backend_cpu):
    if False:
        for i in range(10):
            print('nop')
    dims = fargs_tests[0]
    nc = backend_cpu
    dtype = nc.default_dtype
    randA = rand_unif(dtype, dims)
    randB = rand_unif(dtype, dims)
    compare_helper_mkl('+', randA, randB, nc, dtype)
    compare_helper_mkl('-', randA, randB, nc, dtype)
    compare_helper_mkl('*', randA, randB, nc, dtype)
    compare_helper_mkl('>', randA, randB, nc, dtype)
    compare_helper_mkl('>=', randA, randB, nc, dtype)
    compare_helper_mkl('<', randA, randB, nc, dtype)
    compare_helper_mkl('<=', randA, randB, nc, dtype)

@pytest.mark.hasgpu
def test_math_gpu(fargs_tests, backend_pair_dtype):
    if False:
        return 10
    dims = fargs_tests[0]
    (ng, nc) = backend_pair_dtype
    dtype = ng.default_dtype
    randA = rand_unif(dtype, dims)
    randB = rand_unif(dtype, dims)
    compare_helper('+', randA, randB, ng, nc, dtype)
    compare_helper('-', randA, randB, ng, nc, dtype)
    compare_helper('*', randA, randB, ng, nc, dtype)
    compare_helper('>', randA, randB, ng, nc, dtype)
    compare_helper('>=', randA, randB, ng, nc, dtype)
    compare_helper('<', randA, randB, ng, nc, dtype)
    compare_helper('<=', randA, randB, ng, nc, dtype)

@pytest.mark.hasgpu
def test_slicing(fargs_tests, backend_pair_dtype):
    if False:
        i = 10
        return i + 15
    dims = fargs_tests[0]
    (gpu, cpu) = backend_pair_dtype
    dtype = gpu.default_dtype
    array_np = np.random.uniform(-1, 1, dims).astype(dtype)
    array_ng = gpu.array(array_np, dtype=dtype)
    array_nc = cpu.array(array_np, dtype=dtype)
    assert tensors_allclose(array_ng[0], array_nc[0], rtol=0, atol=0.001)
    assert tensors_allclose(array_ng[-1], array_nc[-1], rtol=0, atol=0.001)
    assert tensors_allclose(array_ng[0, :], array_nc[0, :], rtol=0, atol=0.001)
    assert tensors_allclose(array_ng[0:], array_nc[0:], rtol=0, atol=0.001)
    assert tensors_allclose(array_ng[:-1], array_nc[:-1], rtol=0, atol=0.001)
    assert tensors_allclose(array_ng[:, 0], array_nc[:, 0], rtol=0, atol=0.001)
    assert tensors_allclose(array_ng[:, 0:1], array_nc[:, 0:1], rtol=0, atol=0.001)
    assert tensors_allclose(array_ng[-1, 0:], array_nc[-1:, 0:], rtol=0, atol=0.001)
    array_ng[0] = 0
    array_nc[0] = 0
    assert tensors_allclose(array_ng, array_nc, rtol=0, atol=0.001)

@pytest.mark.hasgpu
def test_reshape_separate(fargs_tests, backend_pair_dtype):
    if False:
        return 10
    dims = fargs_tests[0]
    (gpu, cpu) = backend_pair_dtype
    dtype = gpu.default_dtype
    array_np = np.random.uniform(-1, 1, dims).astype(dtype)
    array_ng = gpu.array(array_np, dtype=dtype)
    array_nc = cpu.array(array_np, dtype=dtype)
    assert array_ng.is_contiguous
    if dims[0] % 2 == 0:
        reshaped_ng = array_ng.reshape((2, dims[0] // 2, dims[1]))
        reshaped_nc = array_nc.reshape((2, dims[0] // 2, dims[1]))
        assert tensors_allclose(reshaped_ng, reshaped_nc, rtol=0, atol=1e-06)

@pytest.mark.hasgpu
def test_reshape_combine(fargs_tests, backend_pair_dtype):
    if False:
        i = 10
        return i + 15
    dims = fargs_tests[0]
    (gpu, cpu) = backend_pair_dtype
    dtype = gpu.default_dtype
    if dims[0] % 2 == 0:
        orig_shape = (2, dims[0] // 2, dims[1])
        array_np = np.random.uniform(-1, 1, orig_shape).astype(dtype)
        array_ng = gpu.array(array_np, dtype=dtype)
        array_nc = cpu.array(array_np, dtype=dtype)
        assert array_ng.is_contiguous
        reshaped_ng = array_ng.reshape(dims)
        reshaped_nc = array_nc.reshape(dims)
        assert tensors_allclose(reshaped_ng, reshaped_nc, rtol=0, atol=1e-06)