import functools
import operator
import unittest
import numpy
import pytest
import six
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv
from chainer_tests.functions_tests.pooling_tests import pooling_nd_helper

@testing.parameterize(*testing.product({'dims': [(4,), (4, 3), (4, 3, 2), (1, 1, 1, 1)], 'dtype': [numpy.float16, numpy.float32, numpy.float64], 'pad_value': [None, 0], 'contiguous': ['C', None]}))
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always']}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}])
class TestAveragePoolingND(testing.FunctionTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim
        self.input_shape = (2, 3) + self.dims
        outs = tuple((conv.get_conv_outsize(d, k, s, p, False) for (d, k, s, p) in six.moves.zip(self.dims, self.ksize, self.stride, self.pad)))
        self.output_shape = (2, 3) + outs
        self.check_backward_options.update({'atol': 0.005, 'rtol': 0.005})
        self.check_double_backward_options.update({'atol': 0.005, 'rtol': 0.005})
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 0.0005, 'rtol': 0.005})
            self.check_backward_options.update({'eps': 0.01, 'atol': 0.005, 'rtol': 0.05})
            self.check_backward_options.update({'eps': 0.01, 'atol': 0.005, 'rtol': 0.05})

    def generate_inputs(self):
        if False:
            while True:
                i = 10
        return (numpy.random.uniform(-1, 1, self.input_shape).astype(self.dtype),)

    def forward(self, inputs, device):
        if False:
            return 10
        (x,) = inputs
        return (functions.average_pooling_nd(x, self.ksize, self.stride, self.pad, self.pad_value),)

    def forward_expected(self, inputs):
        if False:
            print('Hello World!')
        (x,) = inputs
        patches = pooling_nd_helper.pooling_patches(self.dims, self.ksize, self.stride, self.pad, False)

        def denom(idx):
            if False:
                for i in range(10):
                    print('nop')
            if self.pad_value is None:
                s = 1
                for slic in idx:
                    s *= slic.stop - slic.start
                return s
            else:
                return functools.reduce(operator.mul, self.ksize)
        y = []
        for k in six.moves.range(2):
            tmp = []
            for c in six.moves.range(3):
                x_ = x[k, c]
                expect = numpy.array([x_[idx].sum() / denom(idx) for idx in patches])
                expect = expect.reshape(self.output_shape[2:])
                tmp.append(expect)
            y.append(tmp)
        return (numpy.asarray(y, dtype=self.dtype),)

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.inject_backend_tests(['test_forward_consistency', 'test_backward_consistency'], [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always']}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0']}))
class TestConsistencyAveragePoolingND(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        x_shape = (2, 3, 4, 3)
        self.ksize = (3, 3)
        self.stride = (2, 2)
        self.pad = (1, 1)
        self.pad_value = 0
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        outs = tuple((conv.get_conv_outsize(d, k, s, p, False) for (d, k, s, p) in six.moves.zip(x_shape[2:], self.ksize, self.stride, self.pad)))
        gy_shape = (2, 3) + outs
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)
        self.tolerance = {}
        if self.dtype == numpy.float16:
            self.tolerance.update({'atol': 0.001, 'rtol': 0.0001})

    def check_forward_consistency_regression(self, x_data, backend_config):
        if False:
            i = 10
            return i + 15
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        pad_value = self.pad_value
        with backend_config:
            y_nd = functions.average_pooling_nd(x_data, ksize, stride=stride, pad=pad, pad_value=pad_value)
            y_2d = functions.average_pooling_2d(x_data, ksize, stride=stride, pad=pad)
        testing.assert_allclose(y_nd.array, y_2d.array, **self.tolerance)

    def test_forward_consistency(self, backend_config):
        if False:
            print('Hello World!')
        x = self.x.copy()
        x = backend_config.get_array(x)
        self.check_forward_consistency_regression(x, backend_config)

    def check_backward_consistency_regression(self, x_data, gy_data, backend_config):
        if False:
            return 10
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        pad_value = self.pad_value
        x_nd = chainer.Variable(x_data)
        with backend_config:
            y_nd = functions.average_pooling_nd(x_nd, ksize, stride=stride, pad=pad, pad_value=pad_value)
        y_nd.grad = gy_data
        y_nd.backward()
        x_2d = chainer.Variable(x_data)
        with backend_config:
            y_2d = functions.average_pooling_2d(x_2d, ksize, stride=stride, pad=pad)
        y_2d.grad = gy_data
        y_2d.backward()
        testing.assert_allclose(x_nd.grad, x_2d.grad, **self.tolerance)

    def test_backward_consistency(self, backend_config):
        if False:
            i = 10
            return i + 15
        x = backend_config.get_array(self.x)
        gy = backend_config.get_array(self.gy)
        self.check_backward_consistency_regression(x, gy, backend_config)

@testing.parameterize(*testing.product({'dims': [(4, 3, 2), (3, 2), (2,)], 'use_cudnn': ['always', 'auto', 'never'], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@attr.cudnn
class TestAveragePoolingNDCudnnCall(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim
        x_shape = (2, 3) + self.dims
        self.x = cuda.cupy.arange(functools.reduce(operator.mul, x_shape), dtype=self.dtype).reshape(x_shape)
        gy_shape = (2, 3) + tuple((conv.get_conv_outsize(d, k, s, p) for (d, k, s, p) in six.moves.zip(self.dims, self.ksize, self.stride, self.pad)))
        self.gy = cuda.cupy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

    def forward(self):
        if False:
            while True:
                i = 10
        x = chainer.Variable(self.x)
        return functions.average_pooling_nd(x, self.ksize, self.stride, self.pad)

    def test_call_cudnn_forward(self):
        if False:
            while True:
                i = 10
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.pooling_forward') as func:
                self.forward()
                assert func.called == (chainer.should_use_cudnn('>=auto') and self.ndim > 1)

    def test_call_cudnn_backward(self):
        if False:
            i = 10
            return i + 15
        with chainer.using_config('use_cudnn', self.use_cudnn):
            expect = chainer.should_use_cudnn('>=auto') and self.ndim > 1
            y = self.forward()
        y.grad = self.gy
        with testing.patch('cupy.cudnn.pooling_backward') as func:
            y.backward()
            assert func.called == expect

class TestAveragePoolingNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        if False:
            while True:
                i = 10
        x_shape = (2, 3) + (3,) * ndim
        dtype = numpy.float32
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        ksize = (2,) * ndim
        return (x, ksize)

    def test_average_pooling_1d(self):
        if False:
            while True:
                i = 10
        (x, ksize) = self._get_data(1)
        testing.assert_allclose(functions.average_pooling_nd(x, ksize).array, functions.average_pooling_1d(x, ksize).array)

    def test_average_pooling_1d_invalid(self):
        if False:
            while True:
                i = 10
        (x, ksize) = self._get_data(2)
        with pytest.raises(ValueError):
            functions.average_pooling_1d(x, ksize)

    def test_average_pooling_3d(self):
        if False:
            return 10
        (x, ksize) = self._get_data(3)
        testing.assert_allclose(functions.average_pooling_nd(x, ksize).data, functions.average_pooling_3d(x, ksize).data)

    def test_average_pooling_3d_invalid(self):
        if False:
            print('Hello World!')
        (x, ksize) = self._get_data(2)
        with pytest.raises(ValueError):
            functions.average_pooling_3d(x, ksize)
testing.run_module(__name__, __file__)