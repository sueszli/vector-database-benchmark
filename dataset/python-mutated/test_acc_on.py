"""
Test of the acc_on functionality
"""
from utils import tensors_allclose, allclose_with_out
import numpy as np
import pytest
from neon import NervanaObject
from neon.initializers.initializer import Uniform
from neon.layers.layer import Convolution, Deconvolution, Layer, Linear, Bias, BatchNorm
from neon.layers.container import Sequential

def pytest_generate_tests(metafunc):
    if False:
        return 10
    if 'test_args' in metafunc.fixturenames:
        batch_size = 64
        indim = 16
        nifm = 4
        fshape = 2
        nofm = 16
        stride = 1
        pad = 0
        stride = 1
        init_unif = Uniform(low=0.0, high=1.0)
        fargs = [[indim, nifm, fshape, nofm, batch_size, stride, pad, init_unif]]
        metafunc.parametrize('test_args', fargs)

def layer_setup(layer, in_shape, deltas_buffer):
    if False:
        i = 10
        return i + 15
    '\n    Generic layer setup\n    '
    layer.configure(in_shape)
    layer.prev_layer = True
    with pytest.raises(BufferError):
        layer.set_acc_on(True)
    layer.allocate(accumulate_updates=True)
    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

def container_setup(layer, in_shape, deltas_buffer):
    if False:
        print('Hello World!')
    '\n    Generic layer setup\n    '
    layer.configure([in_shape, in_shape])
    layer.prev_layer = True
    with pytest.raises(BufferError):
        layer.set_acc_on(True)
    layer.allocate(accumulate_updates=True)
    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

def random_fprop_layer(in_shape, in_size, layer):
    if False:
        while True:
            i = 10
    dtypeu = np.float32
    inp_rng = [0.0, 1.0]
    inpa = np.random.random(in_size)
    inpa *= inp_rng[1] - inp_rng[0]
    inpa += inp_rng[0]
    inpa = inpa.astype(dtypeu)
    inp = layer.be.array(inpa)
    inp.lshape = in_shape
    return layer.fprop(inp).get().shape

def errfn(layer, out_shape):
    if False:
        i = 10
        return i + 15
    '\n    Random errors (deltas from a previous layer) used for bprop testing\n    '
    dtypeu = np.float32
    erra = np.random.random(out_shape)
    erra = erra.astype(dtypeu)
    err = layer.be.array(erra)
    return err

def check_acc_on(layer, out_shape, buffers):
    if False:
        for i in range(10):
            print('nop')
    err1 = errfn(layer, out_shape)
    err2 = errfn(layer, out_shape)
    assert not tensors_allclose(err1, err2, atol=0.0, rtol=0.0)
    terminal_layer = layer.get_terminal()
    for b in buffers:
        assert getattr(terminal_layer, b).get().shape == getattr(terminal_layer, 'acc_' + b).get().shape
        layer.bprop(err1)
        dW_s = getattr(terminal_layer, b).get()
        layer.bprop(err2)
        dW = getattr(terminal_layer, b).get()
        layer.set_acc_on(True)
        assert hasattr(terminal_layer, 'acc_' + b)
        layer.bprop(err1)
        dW_p = getattr(terminal_layer, b).get()
        assert allclose_with_out(dW_p, dW + dW_s)
        layer.set_acc_on(False)
        layer.bprop(err2)
        dW_dp = getattr(terminal_layer, b).get()
        assert allclose_with_out(dW_dp, dW)

def test_unsupported_layer():
    if False:
        return 10
    layer = Layer()
    with pytest.raises(TypeError):
        layer.allocate(accumulate_updates=True)
    with pytest.raises(BufferError):
        layer.set_acc_on(True)

def test_unsupported_class():
    if False:
        print('Hello World!')
    " Test that the decorator doesn't work if applied\n        to a non-Layer subclass\n    "

    class Foo:

        @Layer.accumulates
        def bprop(self):
            if False:
                print('Hello World!')
            pass
    layer = Foo()
    with pytest.raises(TypeError):
        layer.bprop()

def test_api(backend_default, mocker):
    if False:
        for i in range(10):
            print('nop')
    ' Basic test for API breakage, not working as intended\n        with `self.p = MyTensor(np.ones(10))`, but this could\n        be due to invalid use of something.'

    class MyTensor(object):

        def __init__(self, arr):
            if False:
                i = 10
                return i + 15
            self.arr = arr
            self.shape = arr.shape

        def __add__(self, x):
            if False:
                i = 10
                return i + 15
            return MyTensor(self.get() + x.get())

        def __setitem__(self, index, value):
            if False:
                return 10
            self.arr[index] = value.get()

        def __getitem__(self, index):
            if False:
                return 10
            return self.arr[index]

        def get(self):
            if False:
                return 10
            return self.arr

    def my_get(self):
        if False:
            while True:
                i = 10
        return self

    def my_allocate(self, accumulate_updates=False):
        if False:
            while True:
                i = 10
        self.accumulate_updates = accumulate_updates
        self.p = MyTensor(np.zeros(10))
        self.acc_p = MyTensor(np.zeros(10))
        self.acc_params = [(self.acc_p, self.p)]

    @Layer.accumulates
    def my_bprop(self, err):
        if False:
            i = 10
            return i + 15
        pass
    layer = Layer()
    mocker.patch('neon.layers.layer.Layer.allocate', my_allocate)
    mocker.patch('neon.layers.layer.Layer.bprop', my_bprop)
    layer.allocate(accumulate_updates=True)
    layer.set_acc_on(True)
    layer.bprop(np.zeros(10))
    check_acc_on(layer, 10, ['p'])

def test_conv_acc_on(backend_default, test_args, deltas_buffer):
    if False:
        print('Hello World!')
    (indim, nifm, fshape, nofm, batch_size, stride, pad, init_unif) = test_args
    in_shape = (nifm, indim, indim)
    layer = Convolution(fshape=(fshape, fshape, nofm), strides=stride, padding=pad, init=init_unif)
    testLayer = LayerTest(layer, in_shape, batch_size, deltas_buffer)
    testLayer.test(['dW'])

def test_deconv_acc_on(backend_default, test_args, deltas_buffer):
    if False:
        return 10
    (indim, nifm, fshape, nofm, batch_size, stride, pad, init_unif) = test_args
    in_shape = (nifm, indim, indim)
    layer = Deconvolution(fshape=(fshape, fshape, nofm), strides=stride, padding=0, init=init_unif)
    testLayer = LayerTest(layer, in_shape, batch_size, deltas_buffer)
    testLayer.test(['dW'])

def test_linear_acc_on(backend_default, test_args, deltas_buffer):
    if False:
        while True:
            i = 10
    (indim, nifm, fshape, nofm, batch_size, stride, pad, init_unif) = test_args
    layer = Linear(nout=indim, init=init_unif)
    in_shape = indim
    testLayer = LayerTest(layer, in_shape, batch_size, deltas_buffer)
    testLayer.test(['dW'])

def test_bias_acc_on(backend_default, test_args, deltas_buffer):
    if False:
        return 10
    (indim, nifm, fshape, nofm, batch_size, stride, pad, init_unif) = test_args
    layer = Bias(init=init_unif)
    in_shape = (indim, batch_size)
    testLayer = LayerTest(layer, in_shape, batch_size, deltas_buffer)
    testLayer.test(['dW'])

def test_batchnorm_acc_on(backend_default, test_args, deltas_buffer):
    if False:
        while True:
            i = 10
    (indim, nifm, fshape, nofm, batch_size, stride, pad, init_unif) = test_args
    layer = BatchNorm()
    in_shape = (indim, indim)
    testLayer = LayerTest(layer, in_shape, batch_size, deltas_buffer)
    testLayer.test(['grad_beta', 'grad_gamma'])

def test_layer_container(backend_default, test_args, deltas_buffer):
    if False:
        while True:
            i = 10
    (indim, nifm, fshape, nofm, batch_size, stride, pad, init_unif) = test_args
    in_shape = indim
    containerTest = LayerTest(Sequential([Linear(nout=indim, init=init_unif), Linear(nout=indim, init=init_unif)]), in_shape, batch_size, deltas_buffer)
    containerTest.test(['dW'])

def test_layer_container_unsupported_layer(backend_default, test_args, deltas_buffer):
    if False:
        while True:
            i = 10
    (indim, nifm, fshape, nofm, batch_size, stride, pad, init_unif) = test_args
    in_shape = indim

    def fail_on_accumulate_updates(f):
        if False:
            for i in range(10):
                print('nop')

        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            if 'accumulate_updates' in kwargs:
                raise AttributeError('Should not have gotten accumulate updates')
            out = f(*args, **kwargs)
            return out
        return wrapper
    unsupportedLayer = Linear(nout=indim, init=init_unif)
    unsupportedLayer.allocate = fail_on_accumulate_updates(unsupportedLayer.allocate)
    containerTest = LayerTest(Sequential([Linear(nout=indim, init=init_unif), unsupportedLayer, Linear(nout=indim, init=init_unif)]), in_shape, batch_size, deltas_buffer)
    containerTest.test(['dW'])

class LayerTest(object):

    def __init__(self, layer, in_shape, batch_size, deltas_buffer):
        if False:
            for i in range(10):
                print('nop')
        self.layer = layer
        self.in_shape = in_shape
        self.in_size = (np.prod(in_shape), batch_size)
        NervanaObject.be.bsz = batch_size
        layer_setup(self.layer, self.in_shape, deltas_buffer)

    def test(self, buffers):
        if False:
            for i in range(10):
                print('nop')
        self.out_shape = random_fprop_layer(self.in_shape, self.in_size, self.layer)
        check_acc_on(self.layer, self.out_shape, buffers)