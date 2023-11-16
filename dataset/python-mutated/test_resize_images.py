import unittest
import numpy
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

@testing.parameterize(*testing.product({'in_shape': [(2, 3, 8, 6), (2, 1, 4, 6)], 'mode': ['bilinear', 'nearest'], 'align_corners': [True, False]}))
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestResizeImagesForwardIdentity(testing.FunctionTestCase):

    def generate_inputs(self):
        if False:
            return 10
        x = numpy.random.uniform(size=self.in_shape).astype(numpy.float32)
        return (x,)

    def forward_expected(self, inputs):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        return (x,)

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        output_shape = self.in_shape[2:]
        y = functions.resize_images(x, output_shape, mode=self.mode, align_corners=self.align_corners)
        return (y,)

@testing.parameterize(*testing.product({'in_shape': [(2, 2, 4, 4)], 'output_shape': [(2, 2, 2, 2)], 'mode': ['bilinear', 'nearest'], 'align_corners': [True, False]}))
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestResizeImagesForwardDownScale(testing.FunctionTestCase):

    def generate_inputs(self):
        if False:
            i = 10
            return i + 15
        x = numpy.zeros(self.in_shape, dtype=numpy.float32)
        x[:, :, :2, :2] = 1
        x[:, :, 2:, :2] = 2
        x[:, :, :2, 2:] = 3
        x[:, :, 2:, 2:] = 4
        return (x,)

    def forward_expected(self, inputs):
        if False:
            print('Hello World!')
        y_expect = numpy.zeros(self.output_shape, dtype=numpy.float32)
        y_expect[:, :, 0, 0] = 1
        y_expect[:, :, 1, 0] = 2
        y_expect[:, :, 0, 1] = 3
        y_expect[:, :, 1, 1] = (4,)
        return (y_expect,)

    def forward(self, inputs, device):
        if False:
            print('Hello World!')
        (x,) = inputs
        output_shape = self.output_shape[2:]
        y = functions.resize_images(x, output_shape, mode=self.mode, align_corners=self.align_corners)
        return (y,)

@testing.parameterize(*testing.product({'in_shape': [(1, 1, 2, 2)], 'output_shape': [(1, 1, 3, 3)], 'align_corners': [True, False]}))
@testing.inject_backend_tests(None, [{}] + testing.product({'use_cuda': [True], 'use_cudnn': ['never', 'always'], 'cuda_device': [0, 1]}) + testing.product({'use_chainerx': [True], 'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']}))
class TestResizeImagesForwardUpScaleBilnear(testing.FunctionTestCase):

    def generate_inputs(self):
        if False:
            i = 10
            return i + 15
        x = numpy.zeros(self.in_shape, dtype=numpy.float32)
        x[:, :, 0, 0] = 1
        x[:, :, 1, 0] = 2
        x[:, :, 0, 1] = 3
        x[:, :, 1, 1] = 4
        return (x,)

    def forward_expected(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        y_expect = numpy.zeros(self.output_shape, dtype=numpy.float32)
        y_expect[0, 0, :, :] = numpy.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]], dtype=numpy.float32)
        return (y_expect,)

    def forward(self, inputs, device):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        output_shape = self.output_shape[2:]
        y = functions.resize_images(x, output_shape, align_corners=self.align_corners)
        return (y,)

class TestResizeImagesForwardMultiLinesAlignCorners(unittest.TestCase):
    in_shape = (1, 1, 987, 123)
    output_shape = (1, 1, 765, 345)

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = numpy.arange(numpy.prod(self.in_shape), dtype=numpy.float32)
        self.x = self.x.reshape(self.in_shape)
        out_row = numpy.linspace(0, 123 - 1, 345, dtype=numpy.float32)
        out_col = numpy.linspace(0, (987 - 1) * 123, 765, dtype=numpy.float32)
        self.out = (out_row + out_col[:, None]).reshape(self.output_shape)

    def check_forward(self, x, output_shape):
        if False:
            while True:
                i = 10
        y = functions.resize_images(x, output_shape)
        testing.assert_allclose(y.data, self.out)

    def test_forward_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_forward(self.x, output_shape=self.output_shape[2:])

    @attr.gpu
    def test_forward_gpu(self):
        if False:
            print('Hello World!')
        self.check_forward(cuda.to_gpu(self.x), output_shape=self.output_shape[2:])

@testing.parameterize(*testing.product({'in_shape': [(2, 3, 8, 6), (2, 1, 4, 6)], 'output_shape': [(10, 5), (3, 4)], 'mode': ['bilinear', 'nearest'], 'align_corners': [False, True]}))
class TestResizeImagesBackward(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = numpy.random.uniform(size=self.in_shape).astype(numpy.float32)
        output_shape_4d = self.in_shape[:2] + self.output_shape
        self.gy = numpy.random.uniform(size=output_shape_4d).astype(numpy.float32)
        self.ggx = numpy.random.uniform(size=self.in_shape).astype(numpy.float32)

    def check_backward(self, x, output_shape, gy):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                return 10
            return functions.resize_images(x, output_shape, mode=self.mode, align_corners=self.align_corners)
        gradient_check.check_backward(f, x, gy, dtype='d', atol=0.01, rtol=0.001, eps=1e-05)

    def test_backward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_backward(self.x, self.output_shape, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            return 10
        self.check_backward(cuda.to_gpu(self.x), self.output_shape, cuda.to_gpu(self.gy))

    def check_double_backward(self, x, output_shape, gy, ggx):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            return functions.resize_images(x, output_shape, mode=self.mode, align_corners=self.align_corners)
        gradient_check.check_double_backward(f, x, gy, ggx, atol=0.01, rtol=0.001)

    def test_double_backward_cpu(self):
        if False:
            print('Hello World!')
        self.check_double_backward(self.x, self.output_shape, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_double_backward(cuda.to_gpu(self.x), self.output_shape, cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))
testing.run_module(__name__, __file__)