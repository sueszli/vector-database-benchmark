import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr

class TestInception(unittest.TestCase):
    in_channels = 3
    (out1, proj3, out3, proj5, out5, proj_pool) = (3, 2, 3, 2, 3, 3)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = numpy.random.uniform(-1, 1, (10, self.in_channels, 5, 5)).astype(numpy.float32)
        out = self.out1 + self.out3 + self.out5 + self.proj_pool
        self.gy = numpy.random.uniform(-1, 1, (10, out, 5, 5)).astype(numpy.float32)
        self.l = links.Inception(self.in_channels, self.out1, self.proj3, self.out3, self.proj5, self.out5, self.proj_pool)

    def check_backward(self, x_data, y_grad):
        if False:
            print('Hello World!')
        x = chainer.Variable(x_data)
        y = self.l(x)
        y.grad = y_grad
        y.backward()

    def test_backward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            i = 10
            return i + 15
        with testing.assert_warns(DeprecationWarning):
            self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))
testing.run_module(__name__, __file__)