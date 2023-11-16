"""
test batched_dot behaviors between NervanaCPU, NervanaMKL, and NervanaGPU backend
against numpy.
In NervanaGPU, it supports both N as inside dimension or as outer dimension.
In NervanaCPU, it only supports N as inside dimension, since this is what we use.
In NervanaMKL, it supports both N as inside dimension or as outer dimension.
"""
from __future__ import print_function
import numpy as np
import pytest
from utils import tensors_allclose
size = 32

def setup_test_data(X, N, C, K, dtype):
    if False:
        print('Hello World!')
    dimW = (K, C)
    dimI = (X, C, N)
    dimO = (X, K, N)
    cpuI = np.random.uniform(-1.0, 1.0, dimI).astype(dtype)
    cpuE = np.random.uniform(-1.0, 1.0, dimO).astype(dtype)
    cpuW = np.random.uniform(-1.0, 1.0, dimW).astype(dtype)
    return (cpuI, cpuE, cpuW)

def run_batched_dot(lib, I, E, W, X, dtype):
    if False:
        while True:
            i = 10
    devI = lib.array(I, dtype=dtype)
    devE = lib.array(E, dtype=dtype)
    devW = lib.array(W, dtype=dtype)
    devO = lib.zeros(E.shape, dtype=dtype)
    devB = lib.zeros(I.shape, dtype=dtype)
    devU = lib.zeros(W.shape, dtype=dtype)
    if lib.__class__.__name__.endswith('CPU') | lib.__class__.__name__.endswith('MKL'):
        lib.batched_dot(devW, devI, devO)
        lib.batched_dot(devW.T, devE, devB)
        lib.batched_dot(devE, devI.T, devU)
    elif lib.__class__.__name__.endswith('GPU'):
        lib.batched_dot(devW, devI, devO, size=size)
        lib.batched_dot(devW.T, devE, devB, size=size)
        lib.batched_dot(devE, devI.T, devU, size=size)
    else:
        for i in range(X):
            devO[i] = np.dot(W, I[i])
            devB[i] = np.dot(W.T, E[i])
            devU += np.dot(E[i], I[i].T)
    return (devO, devB, devU)

def test_batched_dot_mkl(backend_pair_bench_mkl):
    if False:
        print('Hello World!')
    np.set_printoptions(threshold=8192 * 4, linewidth=600, formatter={'int': lambda x: '%2d' % x, 'float': lambda x: '%2.0f' % x})
    (nm, nc) = backend_pair_bench_mkl
    dtype = np.float32
    X = 100
    N = 32
    C = 1536
    K = 768
    (cpuI, cpuE, cpuW) = setup_test_data(X, N, C, K, dtype)
    (ncO, ncB, ncU) = run_batched_dot(nc, cpuI, cpuE, cpuW, X, dtype)
    (npO, npB, npU) = run_batched_dot(np, cpuI, cpuE, cpuW, X, dtype)
    (nmO, nmB, nmU) = run_batched_dot(nm, cpuI, cpuE, cpuW, X, dtype)
    assert tensors_allclose(npO, nmO, rtol=0, atol=0.001)
    assert tensors_allclose(npB, nmB, rtol=0, atol=0.001)
    assert tensors_allclose(npU, nmU, rtol=0, atol=0.001)
    assert tensors_allclose(npO, ncO, rtol=0, atol=0.001)
    assert tensors_allclose(npB, ncB, rtol=0, atol=0.001)
    assert tensors_allclose(npU, ncU, rtol=0, atol=0.001)

@pytest.mark.hasgpu
def test_batched_dot(backend_pair_bench):
    if False:
        while True:
            i = 10
    np.set_printoptions(threshold=8192 * 4, linewidth=600, formatter={'int': lambda x: '%2d' % x, 'float': lambda x: '%2.0f' % x})
    (ng, nc) = backend_pair_bench
    dtype = np.float32
    X = 100
    N = 32
    C = 1536
    K = 768
    (cpuI, cpuE, cpuW) = setup_test_data(X, N, C, K, dtype)
    (ncO, ncB, ncU) = run_batched_dot(nc, cpuI, cpuE, cpuW, X, dtype)
    (npO, npB, npU) = run_batched_dot(np, cpuI, cpuE, cpuW, X, dtype)
    if ng.compute_capability > (5, 0):
        (ngO, ngB, ngU) = run_batched_dot(ng, cpuI, cpuE, cpuW, X, dtype)
        assert tensors_allclose(npO, ngO, rtol=0, atol=0.001)
        assert tensors_allclose(npB, ngB, rtol=0, atol=0.001)
        assert tensors_allclose(npU, ngU, rtol=0, atol=0.001)
    assert tensors_allclose(npO, ncO, rtol=0, atol=0.001)
    assert tensors_allclose(npB, ncB, rtol=0, atol=0.001)
    assert tensors_allclose(npU, ncU, rtol=0, atol=0.001)