"""
Test of the mlp/linear layer
"""
import itertools as itt
import numpy as np
from neon import NervanaObject
from neon.initializers.initializer import Uniform
from neon.layers.layer import Linear
from utils import allclose_with_out

def pytest_generate_tests(metafunc):
    if False:
        i = 10
        return i + 15
    if metafunc.config.option.all:
        bsz_rng = [16, 32, 64]
    else:
        bsz_rng = [128]
    if 'basic_linargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [1, 2, 1023, 1024, 1025]
            nout_rng = [1, 4, 1023, 1024, 1025]
        else:
            nin_rng = [4, 32]
            nout_rng = [3, 33]
        fargs = itt.product(nin_rng, nout_rng, bsz_rng)
        metafunc.parametrize('basic_linargs', fargs)
    if 'allrand_args' in metafunc.fixturenames:
        fargs = []
        eps = np.finfo(np.float32).eps
        w_rng = [[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]
        if metafunc.config.option.all:
            rng_max = [eps, eps * 10, 1.0, 2048.0, 1000000.0, 10000000000.0]
        else:
            rng_max = [eps, 1.0, 10000000000.0]
        fargs = itt.product(w_rng, rng_max)
        metafunc.parametrize('allrand_args', fargs)

def test_linear_zeros(backend_default, basic_linargs, deltas_buffer):
    if False:
        return 10
    (nin, nout, batch_size) = basic_linargs
    NervanaObject.be.bsz = batch_size
    dtypeu = np.float32
    init_unif = Uniform(low=0.0, high=0.0)
    layer = Linear(nout=nout, init=init_unif)
    inp = layer.be.array(dtypeu(np.random.random((nin, batch_size))))
    layer.configure(nin)
    layer.prev_layer = True
    layer.allocate()
    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)
    out = layer.fprop(inp).get()
    assert np.min(out) == 0.0 and np.max(out) == 0.0
    err = dtypeu(np.zeros((nout, batch_size)))
    deltas = layer.bprop(layer.be.array(err)).get()
    assert np.min(deltas) == 0.0 and np.max(deltas) == 0.0
    dw = layer.dW.get()
    assert np.min(dw) == 0.0 and np.max(dw) == 0.0
    return

def test_linear_ones(backend_default, basic_linargs, deltas_buffer):
    if False:
        return 10
    (nin, nout, batch_size) = basic_linargs
    NervanaObject.be.bsz = batch_size
    dtypeu = np.float32
    init_unif = Uniform(low=1.0, high=1.0)
    layer = Linear(nout=nout, init=init_unif)
    inp = layer.be.array(dtypeu(np.ones((nin, batch_size))))
    layer.configure(nin)
    layer.prev_layer = True
    layer.allocate()
    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)
    out = layer.fprop(inp).get()
    w = layer.W.get()
    sums = np.sum(w, 1).reshape((nout, 1)) * np.ones((1, batch_size))
    assert allclose_with_out(sums, out, atol=0.0, rtol=0.0), '%e' % np.max(np.abs(out - sums))
    return

def test_all_rand(backend_default, allrand_args, deltas_buffer):
    if False:
        i = 10
        return i + 15
    dtypeu = np.float32
    (w_rng, rngmax) = allrand_args
    inp_rng = [0.0, rngmax]
    nin = 1024
    nout = 2048
    batch_size = 16
    NervanaObject.be.bsz = batch_size
    init_unif = Uniform(low=w_rng[0], high=w_rng[1])
    layer = Linear(nout=nout, init=init_unif)
    inp = np.random.random((nin, batch_size))
    inp *= inp_rng[1] - inp_rng[0]
    inp += inp_rng[0]
    inp = inp.astype(dtypeu)
    layer.configure(nin)
    layer.prev_layer = True
    layer.allocate()
    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)
    out = layer.fprop(layer.be.array(inp)).get()
    w = layer.W.get()
    out_exp = np.dot(w, inp)
    atol = 2 * est_mm_prec(w, inp, ntrials=1)
    assert allclose_with_out(out_exp, out, atol=atol, rtol=0.0), '%e %e' % (np.max(np.abs(out - out_exp)), atol)
    err = np.random.random((nout, batch_size))
    err = err * (inp_rng[1] - inp_rng[0]) + inp_rng[0]
    err = err.astype(dtypeu)
    deltas = layer.bprop(layer.be.array(err)).get()
    dw = layer.dW.get()
    deltas_exp = np.dot(w.T, err)
    atol = 2 * est_mm_prec(w.T, err, ntrials=1)
    assert allclose_with_out(deltas_exp, deltas, atol=atol, rtol=0.0), '%e %e' % (np.max(np.abs(deltas_exp - deltas)), atol)
    dw_exp = np.dot(err, inp.T)
    atol = 2 * est_mm_prec(err, inp.T, ntrials=1)
    assert allclose_with_out(dw_exp, dw, atol=atol, rtol=0.0), '%e %e' % (np.max(np.abs(dw_exp - dw)), atol)
    return

def est_mm_prec(A, B, ntrials=1):
    if False:
        i = 10
        return i + 15
    A64 = np.float64(A)
    B64 = np.float64(B)
    gt = np.dot(A64, B64)
    max_err = -1.0
    for trial in range(ntrials):
        inds = np.random.permutation(A.shape[1])
        C = np.dot(A[:, inds], B[inds, :])
        dd = np.float32(gt - C)
        max_err = max(max_err, np.max(np.abs(dd)))
    max_err *= 10.0
    return max_err