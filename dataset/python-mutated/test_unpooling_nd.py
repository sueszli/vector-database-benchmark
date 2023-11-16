import itertools
import unittest
import numpy
import six
import chainer
from chainer import backend
from chainer import functions
from chainer import testing
from chainer.utils import conv
from chainer.utils import type_check

def xs_iter(dims):
    if False:
        i = 10
        return i + 15
    return itertools.product(*[range(d) for d in dims])

def kxs_iter(x, outs, ksize, stride, pad):
    if False:
        while True:
            i = 10
    return itertools.product(*[range(max(0, -p + s * _x), min(-p + s * _x + k, out)) for (_x, out, k, s, p) in zip(x, outs, ksize, stride, pad)])

def expected_unpooling_nd(x_data, outs, ksize, stride, pad):
    if False:
        for i in range(10):
            print('nop')
    (N, c) = x_data.shape[:2]
    dims = x_data.shape[2:]
    y_expected_shape = (N, c) + outs
    y_expected = numpy.zeros(y_expected_shape, dtype=x_data.dtype)
    for i in six.moves.range(N):
        for _c in six.moves.range(c):
            for x in xs_iter(dims):
                x_idx = (i, _c) + x
                for kx in kxs_iter(x, outs, ksize, stride, pad):
                    y_idx = (i, _c) + kx
                    y_expected[y_idx] += x_data[x_idx]
    return y_expected

@testing.parameterize(*testing.product({'dims': [(5,), (2, 3, 4)], '_ksize': [3], '_stride': [3], '_pad': [1], 'cover_all': [True], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}) + testing.product({'dims': [(3, 2)], '_ksize': [1, 2, 3], '_stride': [1, 2, 3], '_pad': [0, 1], 'cover_all': [True, False], 'dtype': [numpy.float32]}))
@testing.inject_backend_tests(['test_forward', 'test_backward', 'test_double_backward', 'test_consistency_regression_forward', 'test_consistency_regression_backward'], testing.product({'use_cuda': [False], 'use_ideep': ['never', 'always']}) + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestUnpoolingND(testing.FunctionTestCase):

    def setUp(self):
        if False:
            return 10
        N = 2
        c = 3
        self.ndim = len(self.dims)
        self.ksize = (self._ksize,) * self.ndim
        self.stride = (self._stride,) * self.ndim
        self.pad = (self._pad,) * self.ndim
        self.x_shape = (N, c) + self.dims
        self.outs = tuple((conv.get_deconv_outsize(d, k, s, p, cover_all=self.cover_all) for (d, k, s, p) in zip(self.dims, self.ksize, self.stride, self.pad)))
        self.gy_shape = (N, c) + self.outs
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 2 ** (-4), 'rtol': 2 ** (-4)}
            self.check_backward_options = {'atol': 2 ** (-4), 'rtol': 2 ** (-4)}
            self.check_double_backward_options = {}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {'atol': 0.001, 'rtol': 0.001}
            self.check_double_backward_options = {'atol': 0.003, 'rtol': 0.03}

    def generate_inputs(self):
        if False:
            i = 10
            return i + 15
        x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        return (x,)

    def forward_expected(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        outs = self.gy_shape[2:]
        y_expected = expected_unpooling_nd(x, outs, self.ksize, self.stride, self.pad)
        return (y_expected,)

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        y = functions.unpooling_nd(x, self.ksize, self.stride, self.pad, cover_all=self.cover_all)
        return (y,)

    def check_forward_consistency_regression(self, backend_config):
        if False:
            i = 10
            return i + 15
        (inputs,) = self.generate_inputs()
        x = chainer.Variable(backend_config.get_array(inputs))
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        y_nd = functions.unpooling_nd(x, ksize, stride=stride, pad=pad, cover_all=self.cover_all)
        y_2d = functions.unpooling_2d(x, ksize, stride=stride, pad=pad, cover_all=self.cover_all)
        testing.assert_allclose(y_nd.array, y_2d.array, **self.check_forward_options)

    def test_consistency_regression_forward(self, backend_config):
        if False:
            while True:
                i = 10
        if len(self.dims) == 2:
            self.check_forward_consistency_regression(backend_config)

    def check_backward_consistency_regression(self, backend_config):
        if False:
            i = 10
            return i + 15
        (x_data,) = self.generate_inputs()
        gy_data = numpy.random.uniform(-1, 1, self.gy_shape).astype(self.dtype)
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        xp = backend.get_array_module(x_data)
        x_nd = chainer.Variable(xp.array(x_data))
        y_nd = functions.unpooling_nd(x_nd, ksize, stride=stride, pad=pad, cover_all=self.cover_all)
        y_nd.grad = gy_data
        y_nd.backward()
        x_2d = chainer.Variable(xp.array(x_data))
        y_2d = functions.unpooling_2d(x_2d, ksize, stride=stride, pad=pad, cover_all=self.cover_all)
        y_2d.grad = gy_data
        y_2d.backward()
        opt = self.check_backward_options
        testing.assert_allclose(x_nd.grad, x_2d.grad, atol=opt['atol'], rtol=opt['rtol'])

    def test_consistency_regression_backward(self, backend_config):
        if False:
            for i in range(10):
                print('nop')
        ndim = len(self.dims)
        if ndim == 2:
            self.check_backward_consistency_regression(backend_config)

@testing.parameterize(*testing.product({'outsize': [(10,), (10, 9), (10, 9, 8)], '_ksize': [1, 2, 3], '_stride': [1, 2, 3], '_pad': [0, 1], 'cover_all': [True, False]}))
class TestUnpoolingNDOutsize(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.N = 2
        self.c = 3
        ndim = len(self.outsize)
        self.ksize = (self._ksize,) * ndim
        self.stride = (self._stride,) * ndim
        self.pad = (self._pad,) * ndim

    def test_valid_insize(self):
        if False:
            i = 10
            return i + 15
        N = self.N
        c = self.c
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        outs = self.outsize
        cover_all = self.cover_all
        dims = tuple((conv.get_conv_outsize(out, k, s, p, cover_all=cover_all) for (out, k, s, p) in zip(outs, ksize, stride, pad)))
        x_shape = (N, c) + dims
        x_data = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        x = chainer.Variable(x_data)
        y = functions.unpooling_nd(x, ksize, stride, pad, outsize=outs, cover_all=cover_all)
        y_expected = expected_unpooling_nd(x_data, outs, ksize, stride, pad)
        testing.assert_allclose(y_expected, y.data)

    def test_invalid_insize(self):
        if False:
            while True:
                i = 10
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        outs = self.outsize
        cover_all = self.cover_all
        dims = tuple((conv.get_conv_outsize(out, k, s, p, cover_all=cover_all) for (out, k, s, p) in zip(outs, ksize, stride, pad)))
        dims = tuple((d + 1 for d in dims))
        x_shape = (self.N, self.c) + dims
        x_data = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        x = chainer.Variable(x_data)
        with self.assertRaises(type_check.InvalidType):
            functions.unpooling_nd(x, ksize, stride, pad, outsize=outs, cover_all=cover_all)

class TestUnpoolingNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        if False:
            return 10
        x_shape = (2, 3) + (3,) * ndim
        dtype = numpy.float32
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        ksize = (2,) * ndim
        return (x, ksize)

    def test_unpooling_1d(self):
        if False:
            i = 10
            return i + 15
        (x, ksize) = self._get_data(1)
        testing.assert_allclose(functions.unpooling_nd(x, ksize).data, functions.unpooling_1d(x, ksize).data)

    def test_unpooling_1d_invalid(self):
        if False:
            i = 10
            return i + 15
        (x, ksize) = self._get_data(2)
        with self.assertRaises(ValueError):
            functions.unpooling_1d(x, ksize)

    def test_unpooling_3d(self):
        if False:
            for i in range(10):
                print('nop')
        (x, ksize) = self._get_data(3)
        testing.assert_allclose(functions.unpooling_nd(x, ksize).data, functions.unpooling_3d(x, ksize).data)

    def test_unpooling_3d_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        (x, ksize) = self._get_data(2)
        with self.assertRaises(ValueError):
            functions.unpooling_3d(x, ksize)
testing.run_module(__name__, __file__)