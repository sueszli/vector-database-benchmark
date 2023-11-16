"""
Generalized gradient testing applied to bias layer
"""
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import Bias
from neon.initializers.initializer import Gaussian
from grad_funcs import general_gradient_comp

class BiasWithReset(Bias):

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.y = None

def pytest_generate_tests(metafunc):
    if False:
        while True:
            i = 10
    if metafunc.config.option.all:
        bsz_rng = [16, 32, 64]
    else:
        bsz_rng = [16]
    if 'biasargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [1, 2, 3, 4, 15, 16, 17, 32]
        else:
            nin_rng = [1, 2, 32]
        fargs = itt.product(nin_rng, bsz_rng)
        metafunc.parametrize('biasargs', fargs)

def test_bias(backend_cpu64, biasargs):
    if False:
        for i in range(10):
            print('nop')
    (n, batch_size) = biasargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    init = Gaussian()
    layer = BiasWithReset(init=init)
    inp = np.random.randn(n, batch_size)
    epsilon = 1e-05
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=inp.shape, pert_inds=pert_inds)
    assert max_abs < 1e-07

@pytest.mark.xfail(reason='Precision differences with MKL backend. #914')
def test_bias_mkl(backend_mkl, biasargs):
    if False:
        while True:
            i = 10
    (n, batch_size) = biasargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    init = Gaussian()
    layer = BiasWithReset(init=init)
    inp = np.random.randn(n, batch_size)
    epsilon = 1e-05
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=inp.shape, pert_inds=pert_inds)
    assert max_abs < 1e-07