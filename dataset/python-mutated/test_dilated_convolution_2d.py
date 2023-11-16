import unittest
import numpy
import six.moves.cPickle as pickle
import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv

class TestDilatedConvolution2D(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.link = links.DilatedConvolution2D(3, 2, 3, stride=2, pad=2, dilate=2)
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.cleargrads()
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 2, 2, 2)).astype(numpy.float32)

    @attr.gpu
    def test_im2col_consistency(self):
        if False:
            print('Hello World!')
        col_cpu = conv.im2col_cpu(self.x, 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        col_gpu = conv.im2col_gpu(cuda.to_gpu(self.x), 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

    @attr.gpu
    def test_col2im_consistency(self):
        if False:
            i = 10
            return i + 15
        col = conv.im2col_cpu(self.x, 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        (h, w) = self.x.shape[2:]
        im_cpu = conv.col2im_cpu(col, 2, 2, 2, 2, h, w, dy=2, dx=2)
        im_gpu = conv.col2im_gpu(cuda.to_gpu(col), 2, 2, 2, 2, h, w, dy=2, dx=2)
        testing.assert_allclose(im_cpu, im_gpu.get())

    def check_forward_consistency(self):
        if False:
            return 10
        x_cpu = chainer.Variable(self.x)
        y_cpu = self.link(x_cpu)
        self.assertEqual(y_cpu.data.dtype, numpy.float32)
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        y_gpu = self.link(x_gpu)
        self.assertEqual(y_gpu.data.dtype, numpy.float32)
        testing.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    def test_forward_consistency(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_forward_consistency()

    @attr.gpu
    def test_forward_consistency_im2col(self):
        if False:
            i = 10
            return i + 15
        with chainer.using_config('use_cudnn', 'never'):
            self.check_forward_consistency()

    def check_backward(self, x_data, y_grad):
        if False:
            return 10
        gradient_check.check_backward(self.link, x_data, y_grad, (self.link.W, self.link.b), eps=0.01, atol=5e-05, rtol=0.0005)

    def test_backward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            return 10
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_im2col(self):
        if False:
            for i in range(10):
                print('nop')
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        with chainer.using_config('use_cudnn', 'never'):
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_pickling(self, x_data):
        if False:
            i = 10
            return i + 15
        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data1 = y.data
        del x, y
        pickled = pickle.dumps(self.link, -1)
        del self.link
        self.link = pickle.loads(pickled)
        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data2 = y.data
        testing.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

    def test_pickling_cpu(self):
        if False:
            return 10
        self.check_pickling(self.x)

    @attr.gpu
    def test_pickling_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_pickling(cuda.to_gpu(self.x))

@testing.parameterize({'args': (2, 3), 'kwargs': {'stride': 2, 'pad': 2, 'dilate': 2}}, {'args': (None, 2, 3), 'kwargs': {'stride': 2, 'pad': 2, 'dilate': 2}})
class TestDilatedConvolution2DParameterShapePlaceholder(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.link = links.DilatedConvolution2D(*self.args, **self.kwargs)
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(numpy.float32)
        self.link(chainer.Variable(self.x))
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.cleargrads()
        self.gy = numpy.random.uniform(-1, 1, (2, 2, 2, 2)).astype(numpy.float32)

    @attr.gpu
    def test_im2col_consistency(self):
        if False:
            while True:
                i = 10
        col_cpu = conv.im2col_cpu(self.x, 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        col_gpu = conv.im2col_gpu(cuda.to_gpu(self.x), 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

    @attr.gpu
    def test_col2im_consistency(self):
        if False:
            i = 10
            return i + 15
        col = conv.im2col_cpu(self.x, 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        (h, w) = self.x.shape[2:]
        im_cpu = conv.col2im_cpu(col, 2, 2, 2, 2, h, w, dy=2, dx=2)
        im_gpu = conv.col2im_gpu(cuda.to_gpu(col), 2, 2, 2, 2, h, w, dy=2, dx=2)
        testing.assert_allclose(im_cpu, im_gpu.get())

    def check_forward_consistency(self):
        if False:
            while True:
                i = 10
        x_cpu = chainer.Variable(self.x)
        y_cpu = self.link(x_cpu)
        self.assertEqual(y_cpu.data.dtype, numpy.float32)
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        y_gpu = self.link(x_gpu)
        self.assertEqual(y_gpu.data.dtype, numpy.float32)
        testing.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    def test_forward_consistency(self):
        if False:
            print('Hello World!')
        self.check_forward_consistency()

    @attr.gpu
    def test_forward_consistency_im2col(self):
        if False:
            print('Hello World!')
        with chainer.using_config('use_cudnn', 'never'):
            self.check_forward_consistency()

    def check_backward(self, x_data, y_grad):
        if False:
            i = 10
            return i + 15
        gradient_check.check_backward(self.link, x_data, y_grad, (self.link.W, self.link.b), eps=0.01, atol=5e-05, rtol=0.0005)

    def test_backward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            while True:
                i = 10
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_im2col(self):
        if False:
            i = 10
            return i + 15
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        with chainer.using_config('use_cudnn', 'never'):
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_pickling(self, x_data):
        if False:
            print('Hello World!')
        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data1 = y.data
        del x, y
        pickled = pickle.dumps(self.link, -1)
        del self.link
        self.link = pickle.loads(pickled)
        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data2 = y.data
        testing.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

    def test_pickling_cpu(self):
        if False:
            return 10
        self.check_pickling(self.x)

    @attr.gpu
    def test_pickling_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_pickling(cuda.to_gpu(self.x))
testing.run_module(__name__, __file__)