"""
Generalized gradient testing applied to dilated conv layer
"""
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import Convolution
from neon.initializers.initializer import Gaussian
from grad_funcs import general_gradient_comp

class ConvWithReset(Convolution):

    def reset(self):
        if False:
            print('Hello World!')
        self.nglayer = None

def pytest_generate_tests(metafunc):
    if False:
        while True:
            i = 10
    if metafunc.config.option.all:
        bsz_rng = [16, 32]
    else:
        bsz_rng = [16]
    if 'convargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [5, 8]
            nifm_rng = [1, 2, 4]
            fs_rng = [2, 3, 4]
            dil_h_rng = [1, 2, 3, 4]
            dil_w_rng = [1, 2, 3, 4]
        else:
            nin_rng = [10]
            nifm_rng = [1, 5]
            fs_rng = [2, 3]
            dil_h_rng = [3]
            dil_w_rng = [3]
        fargs = itt.product(nin_rng, nifm_rng, fs_rng, bsz_rng, dil_h_rng, dil_w_rng)
        metafunc.parametrize('convargs', fargs)

def test_conv(backend_cpu64, convargs):
    if False:
        i = 10
        return i + 15
    (nin, nifm, fside, batch_size, dil_h, dil_w) = convargs
    fshape = (fside, fside, fside)
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1e-05
    inp = np.arange(sz) * 2.5 * epsilon
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))
    lshape = (nifm, nin, nin)
    init = Gaussian()
    layer = ConvWithReset(fshape, strides=2, padding=fside - 1, dilation=dict(dil_d=1, dil_h=dil_h, dil_w=dil_w), init=init)
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=lshape, pert_inds=pert_inds)
    assert max_abs < 1e-07

@pytest.mark.xfail(reason='Precision differences with MKL backend. #914')
def test_conv_mkl(backend_mkl, convargs):
    if False:
        i = 10
        return i + 15
    (nin, nifm, fside, batch_size, dil_h, dil_w) = convargs
    fshape = (fside, fside, fside)
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1e-05
    inp = np.arange(sz) * 2.5 * epsilon
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))
    lshape = (nifm, nin, nin)
    init = Gaussian()
    layer = ConvWithReset(fshape, strides=2, padding=fside - 1, dilation=dict(dil_d=1, dil_h=dil_h, dil_w=dil_w), init=init)
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=lshape, pert_inds=pert_inds)
    assert max_abs < 1e-07