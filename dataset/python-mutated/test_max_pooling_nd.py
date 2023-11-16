import functools
from operator import mul
import unittest
import numpy
import six
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv
from chainer_tests.functions_tests.pooling_tests import pooling_nd_helper

@testing.parameterize(*testing.product({'in_dims': [(4,), (4, 3), (4, 3, 2), (1, 1, 1, 1)], 'cover_all': [True, False], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@testing.inject_backend_tests(None, [{}, {'use_ideep': 'always'}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestMaxPoolingND(testing.FunctionTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.ndim = len(self.in_dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim
        if self.dtype == numpy.float16:
            self.check_backward_options.update({'atol': 0.001, 'rtol': 0.01})
            self.check_double_backward_options.update({'atol': 0.001, 'rtol': 0.01})

    def generate_inputs(self):
        if False:
            i = 10
            return i + 15
        x_shape = (2, 3) + self.in_dims
        if self.test_name in ('test_backward', 'test_double_backward'):
            x = numpy.arange(functools.reduce(mul, x_shape), dtype=self.dtype)
            x = x.reshape(x_shape)
            x = 2 * x / x.size - 1
        else:
            x = numpy.random.randn(*x_shape).astype(self.dtype, copy=False)
        return (x,)

    def forward(self, inputs, device):
        if False:
            for i in range(10):
                print('nop')
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        cover_all = self.cover_all
        (x,) = inputs
        y = functions.max_pooling_nd(x, ksize, stride=stride, pad=pad, cover_all=cover_all)
        return (y,)

    def _get_out_dims(self, in_dims):
        if False:
            print('Hello World!')
        out_dims = tuple((conv.get_conv_outsize(d, k, s, p, self.cover_all) for (d, k, s, p) in six.moves.zip(in_dims, self.ksize, self.stride, self.pad)))
        return out_dims

    def forward_expected(self, inputs):
        if False:
            print('Hello World!')
        in_dims = self.in_dims
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        cover_all = self.cover_all
        patches = pooling_nd_helper.pooling_patches(in_dims, ksize, stride, pad, cover_all)
        (x,) = inputs
        out_dims = self._get_out_dims(x.shape[2:])
        y_shape = x.shape[:2] + out_dims
        x = x.astype(numpy.float64)
        y = numpy.empty(y_shape, numpy.float64)
        for i in six.moves.range(2):
            for c in six.moves.range(3):
                d = numpy.array([x[i, c][idx].max() for idx in patches])
                y[i, c, ...] = d.reshape(out_dims)
        return (y.astype(self.dtype),)

@testing.parameterize(*testing.product({'dims': [(4, 3, 2), (3, 2), (2,)], 'use_cudnn': ['always', 'auto', 'never'], 'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
@attr.cudnn
class TestMaxPoolingNDCudnnCall(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim
        x_shape = (2, 3) + self.dims
        self.x = cuda.cupy.arange(functools.reduce(mul, x_shape), dtype=self.dtype).reshape(x_shape)
        gy_shape = (2, 3) + tuple((conv.get_conv_outsize(d, k, s, p) for (d, k, s, p) in six.moves.zip(self.dims, self.ksize, self.stride, self.pad)))
        self.gy = cuda.cupy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

    def forward(self):
        if False:
            for i in range(10):
                print('nop')
        x = chainer.Variable(self.x)
        return functions.max_pooling_nd(x, self.ksize, self.stride, self.pad, cover_all=False)

    def test_call_cudnn_forward(self):
        if False:
            i = 10
            return i + 15
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.pooling_forward') as func:
                self.forward()
                self.assertEqual(func.called, chainer.should_use_cudnn('>=auto') and self.ndim > 1)

    def test_call_cudnn_backward(self):
        if False:
            return 10
        with chainer.using_config('use_cudnn', self.use_cudnn):
            expect = chainer.should_use_cudnn('>=auto') and self.ndim > 1
            y = self.forward()
        y.grad = self.gy
        with testing.patch('cupy.cudnn.pooling_backward') as func:
            y.backward()
            self.assertEqual(func.called, expect)

class TestMaxPoolingNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        if False:
            return 10
        x_shape = (2, 3) + (3,) * ndim
        dtype = numpy.float32
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        ksize = (2,) * ndim
        return (x, ksize)

    def test_max_pooling_1d(self):
        if False:
            print('Hello World!')
        (x, ksize) = self._get_data(1)
        testing.assert_allclose(functions.max_pooling_nd(x, ksize).data, functions.max_pooling_1d(x, ksize).data)

    def test_max_pooling_1d_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        (x, ksize) = self._get_data(2)
        with self.assertRaises(ValueError):
            functions.max_pooling_1d(x, ksize)

    def test_max_pooling_3d(self):
        if False:
            i = 10
            return i + 15
        (x, ksize) = self._get_data(3)
        testing.assert_allclose(functions.max_pooling_nd(x, ksize).data, functions.max_pooling_3d(x, ksize).data)

    def test_max_pooling_3d_invalid(self):
        if False:
            return 10
        (x, ksize) = self._get_data(2)
        with self.assertRaises(ValueError):
            functions.max_pooling_3d(x, ksize)

class TestMaxPoolingNDIndices(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = numpy.arange(2 * 3 * 4 * 4, dtype=numpy.float32).reshape(2, 3, 4, 4)

    def _check(self, x):
        if False:
            print('Hello World!')
        (out, indices) = functions.max_pooling_nd(x, 2, cover_all=False, return_indices=True)
        assert isinstance(out, chainer.Variable)
        assert isinstance(out.array, type(x))
        assert isinstance(indices, type(x))
        assert indices.shape == out.array.shape
        expect = numpy.zeros(indices.shape, dtype=indices.dtype)
        for i in six.moves.range(2):
            for c in six.moves.range(3):
                xx = x[i, c]
                expect[i, c] = numpy.array([[xx[0:2, 0:2].ravel().argmax(), xx[0:2, 2:4].ravel().argmax()], [xx[2:4, 0:2].ravel().argmax(), xx[2:4, 2:4].ravel().argmax()]])
        if out.xp is cuda.cupy:
            expect = cuda.to_gpu(expect)
        assert (expect == indices).all()

    def test_cpu(self):
        if False:
            return 10
        self._check(self.x)

    @attr.gpu
    @attr.cudnn
    def test_gpu(self):
        if False:
            print('Hello World!')
        x = cuda.to_gpu(self.x)
        with chainer.using_config('use_cudnn', 'never'):
            self._check(x)
        with chainer.using_config('use_cudnn', 'always'):
            self._check(x)
testing.run_module(__name__, __file__)