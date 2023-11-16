"""
Generalized gradient testing applied to mlp/linear layer
"""
from __future__ import print_function
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import Linear
from neon.initializers.initializer import Gaussian
from grad_funcs import general_gradient_comp

class LinearWithReset(Linear):

    def reset(self):
        if False:
            return 10
        pass

def pytest_generate_tests(metafunc):
    if False:
        print('Hello World!')
    if metafunc.config.option.all:
        bsz_rng = [16, 32, 64]
    else:
        bsz_rng = [16]
    if 'mlpargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [1, 2, 3, 10]
            nout_rng = [1, 2, 3, 10]
        else:
            nin_rng = [1, 2]
            nout_rng = [3]
        fargs = itt.product(nin_rng, nout_rng, bsz_rng)
        metafunc.parametrize('mlpargs', fargs)

def test_mlp(backend_cpu64, mlpargs):
    if False:
        for i in range(10):
            print('nop')
    (nin, nout, batch_size) = mlpargs
    batch_size = batch_size
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    init = Gaussian()
    layer = LinearWithReset(nout=nout, init=init)
    inp = np.random.randn(nin, batch_size)
    epsilon = 1e-05
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, pert_inds=pert_inds)
    assert max_abs < 1e-07

@pytest.mark.xfail(reason='Precision differences with MKL backend. #914')
def test_mlp_mkl(backend_mkl, mlpargs):
    if False:
        while True:
            i = 10
    (nin, nout, batch_size) = mlpargs
    batch_size = batch_size
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    init = Gaussian()
    layer = LinearWithReset(nout=nout, init=init)
    inp = np.random.randn(nin, batch_size)
    epsilon = 1e-05
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, pert_inds=pert_inds)
    assert max_abs < 1e-07