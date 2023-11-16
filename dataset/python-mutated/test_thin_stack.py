import unittest
import numpy
import chainer
from chainer import backend
from chainer import cuda
from chainer import testing
from chainer.testing import attr
import thin_stack

class TestThinStackGet(unittest.TestCase):
    shape = (3, 4, 5)
    dtype = numpy.float32

    def setUp(self):
        if False:
            print('Hello World!')
        self.s = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.i = numpy.array([0, 1, 0], numpy.int32)
        x_shape = (len(self.i), self.shape[-1])
        self.gx = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.gt = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def check_forward(self, s_data, i_data):
        if False:
            for i in range(10):
                print('nop')
        xp = backend.get_array_module(s_data)
        s_old = s_data.copy()
        s = chainer.Variable(s_data)
        i = chainer.Variable(i_data)
        (x, t) = thin_stack.thin_stack_get(s, i)
        expect = s_old[xp.arange(len(i_data)), i_data]
        testing.assert_allclose(x.array, expect)
        self.assertIs(s_data, t.array)

    def test_forward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_forward(self.s, self.i)

    @attr.gpu
    def test_forward_gpu(self):
        if False:
            print('Hello World!')
        self.check_forward(cuda.to_gpu(self.s), cuda.to_gpu(self.i))

    def check_backward(self, s_data, i_data, gx_data, gt_data):
        if False:
            i = 10
            return i + 15
        gt_old = gt_data.copy()
        s = chainer.Variable(s_data)
        i = chainer.Variable(i_data)
        (x, t) = thin_stack.thin_stack_get(s, i)
        x.grad = gx_data
        t.grad = gt_data
        t.backward()
        for (j, ind) in enumerate(i_data):
            for k in range(self.shape[1]):
                if k == ind:
                    testing.assert_allclose(s.grad[j, k], gt_old[j, k] + gx_data[j])
                else:
                    testing.assert_allclose(s.grad[j, k], gt_old[j, k])
        self.assertIsNone(i.grad)
        self.assertIs(s.grad, gt_data)

    def test_backward_cpu(self):
        if False:
            return 10
        self.check_backward(self.s, self.i, self.gx, self.gt)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            while True:
                i = 10
        self.check_backward(cuda.to_gpu(self.s), cuda.to_gpu(self.i), cuda.to_gpu(self.gx), cuda.to_gpu(self.gt))

class TestThinStackSet(unittest.TestCase):
    shape = (3, 4, 5)
    dtype = numpy.float32

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.s = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.i = numpy.array([0, 1, 0], numpy.int32)
        x_shape = (len(self.i), self.shape[-1])
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.gt = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def check_forward(self, s_data, i_data, x_data):
        if False:
            i = 10
            return i + 15
        xp = backend.get_array_module(s_data)
        s = chainer.Variable(s_data)
        i = chainer.Variable(i_data)
        x = chainer.Variable(x_data)
        t = thin_stack.thin_stack_set(s, i, x)
        testing.assert_allclose(t.array[xp.arange(len(i_data)), i_data], x_data)
        self.assertIs(s_data, t.array)

    def test_forward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_forward(self.s, self.i, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_forward(cuda.to_gpu(self.s), cuda.to_gpu(self.i), cuda.to_gpu(self.x))

    def check_backward(self, s_data, i_data, x_data, gt_data):
        if False:
            i = 10
            return i + 15
        gt_old = gt_data.copy()
        s = chainer.Variable(s_data)
        i = chainer.Variable(i_data)
        x = chainer.Variable(x_data)
        t = thin_stack.thin_stack_set(s, i, x)
        t.grad = gt_data
        t.backward()
        for (j, ind) in enumerate(i_data):
            testing.assert_allclose(x.grad[j], gt_old[j, ind])
            for k in range(self.shape[1]):
                if k == ind:
                    testing.assert_allclose(s.grad[j, k], 0)
                else:
                    testing.assert_allclose(s.grad[j, k], gt_old[j, k])
        self.assertIsNone(i.grad)
        self.assertIs(s.grad, gt_data)

    def test_backward_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_backward(self.s, self.i, self.x, self.gt)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            print('Hello World!')
        self.check_backward(cuda.to_gpu(self.s), cuda.to_gpu(self.i), cuda.to_gpu(self.x), cuda.to_gpu(self.gt))
testing.run_module(__name__, __file__)