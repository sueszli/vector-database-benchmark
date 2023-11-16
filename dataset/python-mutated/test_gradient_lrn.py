"""
Generalized gradient testing applied to lrn layer
"""
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import LRN
from grad_funcs import general_gradient_comp

class LRNWithReset(LRN):

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.nglayer = None

def pytest_generate_tests(metafunc):
    if False:
        print('Hello World!')
    if metafunc.config.option.all:
        bsz_rng = [8]
    else:
        bsz_rng = [8]
    if 'lrnargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [5, 6]
            nifm_rng = [1, 2, 4]
            fs_rng = [3, 5]
        else:
            nin_rng = [2]
            nifm_rng = [7, 20]
            fs_rng = [3, 5]
        fargs = itt.product(nin_rng, nifm_rng, fs_rng, bsz_rng)
        metafunc.parametrize('lrnargs', fargs)

def test_lrnorm(backend_cpu64, lrnargs):
    if False:
        return 10
    (nin, nifm, fshape, batch_size) = lrnargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1e-05
    inp = np.arange(sz) * 2.5 * epsilon
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))
    lshape = (nifm, nin, nin)
    layer = LRNWithReset(depth=fshape, ascale=0.000125, bpower=0.75)
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=lshape, pert_inds=pert_inds)
    assert max_abs < 1e-06

def test_lrn_large_inp(backend_cpu64, deltas_buffer):
    if False:
        return 10
    nin = 2
    nifm = 16
    depth = 5
    batch_size = 64
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    be = NervanaObject.be
    shape = (nifm * nin * nin, batch_size)
    shape_full = (nifm, nin, nin, batch_size)
    inp_rng = 100000.0
    epsilon = 10.0
    np.random.seed(1234)
    ind_pert = (8, 0, 0, 16)
    ind_pert2 = np.ravel_multi_index(ind_pert[0:3], shape_full[0:3])
    ind_pert = (ind_pert2, ind_pert[-1])
    inp = np.zeros(shape)
    inp[ind_pert] = inp_rng
    inpa = be.array(inp)
    lshape = shape_full[0:3]
    layer = LRNWithReset(depth=depth, ascale=0.000125, bpower=0.75)
    layer.configure(lshape)
    if layer.owns_delta:
        layer.prev_layer = True
    layer.allocate()
    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)
    loss_scale = np.ones(inpa.shape)
    layer.fprop(inpa).get()
    bprop_deltas = layer.bprop(be.array(loss_scale)).get()
    bprop_delta = bprop_deltas[ind_pert]
    inp_p = inp.copy()
    inp_p[ind_pert] += epsilon
    inp_m = inp.copy()
    inp_m[ind_pert] -= epsilon
    out_p = layer.fprop(be.array(inp_p)).get()[ind_pert]
    out_m = layer.fprop(be.array(inp_m)).get()[ind_pert]
    grad_est = 0.5 / float(epsilon) * (out_p - out_m)
    assert np.abs(grad_est - bprop_delta) < 1e-12

@pytest.mark.xfail(reason='Precision differences with MKL backend. #914')
def test_lrnorm_mkl(backend_mkl, lrnargs):
    if False:
        for i in range(10):
            print('nop')
    (nin, nifm, fshape, batch_size) = lrnargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1e-05
    inp = np.arange(sz) * 2.5 * epsilon
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))
    lshape = (nifm, nin, nin)
    layer = LRNWithReset(depth=fshape, ascale=0.000125, bpower=0.75)
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=lshape, pert_inds=pert_inds)
    assert max_abs < 1e-06

@pytest.mark.xfail(reason='Precision differences with MKL backend. #914')
def test_lrn_large_inp_mkl(backend_mkl, deltas_buffer):
    if False:
        for i in range(10):
            print('nop')
    nin = 2
    nifm = 16
    depth = 5
    batch_size = 64
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    be = NervanaObject.be
    shape = (nifm * nin * nin, batch_size)
    shape_full = (nifm, nin, nin, batch_size)
    inp_rng = 100000.0
    epsilon = 10.0
    np.random.seed(1234)
    ind_pert = (8, 0, 0, 16)
    ind_pert2 = np.ravel_multi_index(ind_pert[0:3], shape_full[0:3])
    ind_pert = (ind_pert2, ind_pert[-1])
    inp = np.zeros(shape)
    inp[ind_pert] = inp_rng
    inpa = be.array(inp)
    lshape = shape_full[0:3]
    layer = LRNWithReset(depth=depth, ascale=0.000125, bpower=0.75)
    layer.configure(lshape)
    if layer.owns_delta:
        layer.prev_layer = True
    layer.allocate()
    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)
    loss_scale = np.ones(inpa.shape)
    layer.fprop(inpa).get()
    bprop_deltas = layer.bprop(be.array(loss_scale)).get()
    bprop_delta = bprop_deltas[ind_pert]
    inp_p = inp.copy()
    inp_p[ind_pert] += epsilon
    inp_m = inp.copy()
    inp_m[ind_pert] -= epsilon
    out_p = layer.fprop(be.array(inp_p)).get()[ind_pert]
    out_m = layer.fprop(be.array(inp_m)).get()[ind_pert]
    grad_est = 0.5 / float(epsilon) * (out_p - out_m)
    assert np.abs(grad_est - bprop_delta) < 1e-12