import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers import Deconvolution
from neon.initializers import Uniform
from utils import allclose_with_out
try:
    from neon.backends.nervanagpu import NervanaGPU
except ImportError:

    class NervanaGPU(object):
        pass

def pytest_generate_tests(metafunc):
    if False:
        return 10
    if metafunc.config.option.all:
        bsz_rng = [32, 64]
    else:
        bsz_rng = [32]
    if 'zeros_convargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            fs_rng = [2, 3]
            nfm_rng = [8, 16]
        else:
            fs_rng = [2]
            nfm_rng = [16]
        fargs = itt.product(fs_rng, nfm_rng, bsz_rng)
        metafunc.parametrize('zeros_convargs', fargs)
    if 'ones_convargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            indim_rng = [8, 16]
            nifm_rng = [8, 16, 32]
            fs_rng = [2, 3, 5]
            nofm_rng = [1, 8, 16]
        else:
            indim_rng = [32]
            nifm_rng = [8]
            fs_rng = [2]
            nofm_rng = [16]
        fargs = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng, bsz_rng)
        metafunc.parametrize('ones_convargs', fargs)
    if 'rand_convargs' in metafunc.fixturenames:
        fargs = []
        eps = np.finfo(np.float32).eps
        if metafunc.config.option.all:
            indim_rng = [8, 16]
            nifm_rng = [8, 16]
            fs_rng = [3]
            nofm_rng = [4]
            rng_max_rng = [eps, eps * 10, 1.0, 1000000.0, 10000000000.0]
            wrng = [[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]
        else:
            indim_rng = [16]
            nifm_rng = [8]
            fs_rng = [3]
            nofm_rng = [4]
            rng_max_rng = [1.0, 10.0]
            wrng = [[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]
        fargs = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng, bsz_rng, rng_max_rng, wrng)
        metafunc.parametrize('rand_convargs', fargs)

def test_dconv_zeros(backend_default, zeros_convargs, deltas_buffer):
    if False:
        print('Hello World!')
    (fshape, nofm, batch_size) = zeros_convargs
    NervanaObject.be.bsz = batch_size
    dtypeu = np.float32
    init_unif = Uniform(low=0.0, high=0.0)
    if isinstance(NervanaObject.be, NervanaGPU) and NervanaObject.be.compute_capability < (5, 0):
        inshape = (64, 28, 28)
    else:
        inshape = (64, 28, 28, 28)
    insize = np.prod(inshape)
    neon_layer = Deconvolution(fshape=(fshape, fshape, nofm), strides=1, padding=0, init=init_unif)
    inp_arr_shape = (insize, batch_size)
    inp = np.random.random(inp_arr_shape).astype(dtypeu)
    inp = neon_layer.be.array(inp)
    inp.lshape = inshape
    neon_layer.configure(inshape)
    neon_layer.prev_layer = True
    neon_layer.allocate()
    neon_layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    neon_layer.set_deltas(deltas_buffer)
    outa = neon_layer.fprop(inp)
    out = outa.get()
    assert np.min(out) == 0.0 and np.max(out) == 0.0
    err = dtypeu(np.zeros(outa.shape))
    deltas = neon_layer.bprop(NervanaObject.be.array(err)).get()
    assert np.min(deltas) == 0.0 and np.max(deltas) == 0.0
    dw = neon_layer.dW.get()
    assert np.min(dw) == 0.0 and np.max(dw) == 0.0
    return

def test_dconv_ones(backend_default, ones_convargs, deltas_buffer):
    if False:
        i = 10
        return i + 15
    (indim, nifm, fshape, nofm, batch_size) = ones_convargs
    if isinstance(NervanaObject.be, NervanaGPU) and NervanaObject.be.compute_capability < (5, 0):
        if nofm % 4 != 0:
            pytest.skip(msg='C dim must be a multiple of 4 for Kepler bprop kernel')
    NervanaObject.be.bsz = batch_size
    dtypeu = np.float32
    init_unif = Uniform(low=1.0, high=1.0)
    inshape = (nifm, indim, indim)
    insize = np.prod(inshape)
    neon_layer = Deconvolution(fshape=(fshape, fshape, nofm), strides=1, padding=0, init=init_unif)
    inp = neon_layer.be.array(np.ones((insize, batch_size)).astype(dtypeu))
    inp.lshape = inshape
    neon_layer.configure(inshape)
    neon_layer.prev_layer = True
    neon_layer.allocate()
    neon_layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    neon_layer.set_deltas(deltas_buffer)
    out = neon_layer.fprop(inp).get()
    out_exp_min = nifm
    out_exp_max = fshape * fshape * nifm
    assert np.min(out) == out_exp_min and np.max(out) == out_exp_max
    err = np.ones(out.shape).astype(dtypeu)
    neon_layer.bprop(NervanaObject.be.array(err)).get()
    dw = neon_layer.dW.get()
    ref_layer = DeconvRefLayer(1, batch_size, identity, inshape[0], inshape[1:3], (fshape, fshape), nofm, 1, dtypeu)
    ref_layer.weights = np.ones(neon_layer.W.shape).T.astype(dtypeu)
    ref_layer.bprop(err)
    updates_exp = ref_layer.ofmsize * batch_size
    assert np.max(dw) == updates_exp and np.min(dw) == updates_exp
    assert np.max(np.abs(ref_layer.y.T - neon_layer.deltas.get())) == 0.0
    return

def test_dconv_rand(backend_default, rand_convargs, deltas_buffer):
    if False:
        print('Hello World!')
    (indim, nifm, fshape, nofm, batch_size, rngmax, w_rng) = rand_convargs
    if isinstance(NervanaObject.be, NervanaGPU) and NervanaObject.be.compute_capability < (5, 0):
        if nofm % 4 != 0:
            pytest.skip(msg='C dim must be a multiple of 4 for Kepler bprop kernel')
    NervanaObject.be.bsz = batch_size
    dtypeu = np.float32
    inp_rng = [0.0, rngmax]
    init_unif = Uniform(low=w_rng[0], high=w_rng[1])
    inshape = (indim, indim, nifm)
    insize = np.prod(inshape)
    neon_layer = Deconvolution(fshape=(fshape, fshape, nofm), strides=1, padding=0, init=init_unif)
    insize = np.prod(inshape)
    ref_layer = DeconvRefLayer(1, batch_size, identity, inshape[0], inshape[1:3], (fshape, fshape), nofm, 1, dtypeu)
    inpa = np.random.random((insize, batch_size))
    inpa *= inp_rng[1] - inp_rng[0]
    inpa += inp_rng[0]
    inpa = inpa.astype(dtypeu)
    inp = neon_layer.be.array(inpa)
    inp.lshape = inshape
    neon_layer.configure(inshape)
    neon_layer.prev_layer = True
    neon_layer.allocate()
    neon_out = neon_layer.fprop(inp).get()
    neon_layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    neon_layer.set_deltas(deltas_buffer)
    ref_layer.weights = neon_layer.W.get().T
    ref_out = np.copy(ref_layer.berror)
    ref_layer.fprop(inpa.T, permute=True)
    ref_out2 = ref_layer.berror
    atol = 10 * np.max(np.abs(ref_out - ref_out2))
    assert allclose_with_out(ref_out.T, neon_out, atol=atol, rtol=0.0), '%e %e' % (np.max(np.abs(ref_out.T - neon_out)), atol)
    erra = np.random.random(neon_out.shape)
    erra *= inp_rng[1] - inp_rng[0]
    erra += inp_rng[0]
    erra = erra.astype(dtypeu)
'\nDeconv check code adapted from ref-des\ncnn8 currently only using strides = 1\n'

def identity(x):
    if False:
        while True:
            i = 10
    return x

def identity_prime(x):
    if False:
        print('Hello World!')
    return np.ones(x.shape)

def get_prime(func):
    if False:
        while True:
            i = 10
    if func == identity:
        return identity_prime

class DeconvRefLayer(object):

    def __init__(self, pos, mbs, g, nifm, ifmshape, fshape, nofm, strides, dtypeu):
        if False:
            while True:
                i = 10
        assert g == identity
        self.ofmshape = ifmshape
        (self.ofmheight, self.ofmwidth) = ifmshape
        (self.fheight, self.fwidth) = fshape
        self.ifmheight = (self.ofmheight - 1) * strides + self.fheight
        self.ifmwidth = (self.ofmwidth - 1) * strides + self.fwidth
        self.nifm = nofm
        self.nofm = nifm
        self.ifmshape = (self.ifmheight, self.ifmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.nout = self.ofmsize * self.nofm
        self.fsize = self.nifm * self.fheight * self.fwidth
        self.weights = np.zeros((self.nofm, self.fsize), dtype=dtypeu)
        self.g = g
        self.gprime = get_prime(g)
        self.z = np.zeros((mbs, self.nout), dtype=dtypeu)
        self.y = np.zeros((mbs, self.nout), dtype=dtypeu)
        ofmstarts = np.array(list(range(0, self.ofmsize * self.nofm, self.ofmsize)))
        self.ofmlocs = np.zeros((self.ofmsize, self.nofm), dtype=np.int32)
        for dst in range(self.ofmsize):
            self.ofmlocs[dst, :] = ofmstarts + dst
        self.links = []
        self.makelinks(self.nifm, self.ifmsize, self.ifmshape, self.ofmshape, fshape, strides)
        self.updates = np.zeros(self.weights.shape, dtype=dtypeu)
        self.updateshards = np.zeros((self.fheight * self.fwidth, self.nofm, self.fsize), dtype=dtypeu)
        self.updatebuf = np.zeros((self.nofm, self.fsize), dtype=dtypeu)
        self.pos = pos
        if self.pos > 0:
            self.bpropbuf = np.zeros((mbs, self.fsize), dtype=dtypeu)
            self.berror = np.zeros((mbs, self.ifmsize * self.nifm), dtype=dtypeu)
            self.berrorshards = np.zeros((self.fheight * self.fwidth, mbs, self.ifmsize * self.nifm), dtype=dtypeu)

    def makelinks(self, nifm, ifmsize, ifmshape, ofmshape, fshape, strides):
        if False:
            while True:
                i = 10
        ndims = len(ifmshape)
        dimsizes = np.empty(ndims, dtype='int32')
        for dim in range(ndims):
            dimsizes[dim] = np.prod(ifmshape[dim:])
        links = []
        for ofmdim in np.ndindex(ofmshape):
            src = ofmdim[-1]
            for dim in range(-1, -ndims, -1):
                src += dimsizes[dim] * ofmdim[dim - 1]
            src *= strides
            indlist = list(range(src, src + fshape[-1]))
            for dim in range(-1, -ndims, -1):
                indarray = np.array(indlist)
                for dimind in range(1, fshape[dim - 1]):
                    indlist.extend(list(indarray + dimind * dimsizes[dim]))
            indarray = np.array(indlist)
            for ifm in range(1, nifm):
                indlist.extend(list(indarray + ifm * ifmsize))
            links.append(indlist)
        self.links = np.array(links, dtype='int32')

    def bprop(self, error, permute=False):
        if False:
            return 10
        for dst in range(self.ofmsize):
            rflinks = self.links[dst]
            A = error.T[:, rflinks]
            B = self.weights.T
            if permute:
                inds = np.random.permutation(A.shape[1])
                self.y[:, self.ofmlocs[dst]] = np.dot(A[:, inds], B[inds, :])
            else:
                self.y[:, self.ofmlocs[dst]] = np.dot(A, B)

    def fprop(self, inputs, permute=False):
        if False:
            i = 10
            return i + 15
        if self.pos > 0:
            self.berror.fill(0.0)
            for dst in range(self.ofmsize):
                rflinks = self.links[dst]
                A = inputs[:, self.ofmlocs[dst]]
                B = self.weights
                if permute:
                    inds = np.random.permutation(A.shape[1])
                    np.dot(A[:, inds], B[inds, :], self.bpropbuf)
                else:
                    np.dot(A, B, self.bpropbuf)
                self.berror[:, rflinks] += self.bpropbuf
        self.updates.fill(0.0)
        for dst in range(self.ofmsize):
            rflinks = self.links[dst]
            deltaslice = inputs[:, self.ofmlocs[dst]]
            A = deltaslice.T
            B = inputs[:, rflinks]
            if permute:
                inds = np.random.permutation(A.shape[1])
                np.dot(A[:, inds], B[inds, :], out=self.updatebuf)
            else:
                np.dot(A, B, out=self.updatebuf)
            self.updates += self.updatebuf