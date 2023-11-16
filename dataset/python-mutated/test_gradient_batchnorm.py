"""
Generalized gradient testing applied to batchnorm layer
"""
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import BatchNorm
from grad_funcs import general_gradient_comp

class BatchNormWithReset(BatchNorm):

    def reset(self):
        if False:
            return 10
        self.__init__(rho=self.rho, eps=self.eps, name=self.name)

def pytest_generate_tests(metafunc):
    if False:
        for i in range(10):
            print('nop')
    if metafunc.config.option.all:
        bsz_rng = [16, 32, 64]
    else:
        bsz_rng = [8]
    if 'bnargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            n = [2, 4, 8, 10, 64, (3, 16, 16), (1, 14, 14)]
        else:
            n = [2, 4]
        fargs = itt.product(n, bsz_rng)
        metafunc.parametrize('bnargs', fargs)

def test_batchnorm(backend_cpu64, bnargs):
    if False:
        for i in range(10):
            print('nop')
    (n, batch_size) = bnargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    layer = BatchNormWithReset()
    inp_shape = None
    inp_size = n
    if isinstance(n, tuple):
        inp_shape = n
        inp_size = np.prod(n)
    inp = np.random.randn(inp_size, batch_size)
    epsilon = 1e-05
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=inp_shape, pert_inds=pert_inds)
    assert max_abs < 1e-07

@pytest.mark.xfail(reason='Precision differences with MKL backend. #914')
def test_batchnorm_mkl(backend_mkl, bnargs):
    if False:
        return 10
    (n, batch_size) = bnargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    layer = BatchNormWithReset()
    inp_shape = None
    inp_size = n
    if isinstance(n, tuple):
        inp_shape = n
        inp_size = np.prod(n)
    inp = np.random.randn(inp_size, batch_size)
    epsilon = 1e-05
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=inp_shape, pert_inds=pert_inds)
    assert max_abs < 1e-07