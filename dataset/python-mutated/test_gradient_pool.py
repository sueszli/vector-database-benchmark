"""
Generalized gradient testing applied to pooling layer
"""
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import Pooling
from grad_funcs import general_gradient_comp

class PoolingWithReset(Pooling):

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.nglayer = None

def pytest_generate_tests(metafunc):
    if False:
        for i in range(10):
            print('nop')
    if metafunc.config.option.all:
        bsz_rng = [16, 32]
    else:
        bsz_rng = [16]
    if 'poolargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [5, 8]
            nifm_rng = [1, 2, 4]
            fs_rng = [2, 3, 4]
        else:
            nin_rng = [10]
            nifm_rng = [1, 5]
            fs_rng = [2, 3]
        op_rng = ['max', 'avg']
        fargs = itt.product(nin_rng, nifm_rng, fs_rng, bsz_rng, op_rng)
        metafunc.parametrize('poolargs', fargs)

def test_pooling(backend_cpu64, poolargs):
    if False:
        print('Hello World!')
    (nin, nifm, fshape, batch_size, op) = poolargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1e-05
    inp = np.arange(sz) * 2.5 * epsilon
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))
    lshape = (nifm, nin, nin)
    layer = PoolingWithReset(fshape, op=op)
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=lshape, pert_inds=pert_inds, pooling=True)
    assert max_abs < 1e-07

@pytest.mark.xfail(reason='Known MKL bug. See #913. Also impacted by #914')
def test_pooling_mkl(backend_mkl, poolargs):
    if False:
        return 10
    (nin, nifm, fshape, batch_size, op) = poolargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1e-05
    inp = np.arange(sz) * 2.5 * epsilon
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))
    lshape = (nifm, nin, nin)
    layer = PoolingWithReset(fshape, op=op)
    pert_frac = 0.1
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]
    (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, lshape=lshape, pert_inds=pert_inds, pooling=True)
    assert max_abs < 1e-07