import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr

class TestHighway(unittest.TestCase):
    in_out_size = 3

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = numpy.random.uniform(-1, 1, (5, self.in_out_size)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (5, self.in_out_size)).astype(numpy.float32)
        self.link = links.Highway(self.in_out_size, activate=functions.tanh)
        Wh = self.link.plain.W.data
        Wh[...] = numpy.random.uniform(-1, 1, Wh.shape)
        bh = self.link.plain.b.data
        bh[...] = numpy.random.uniform(-1, 1, bh.shape)
        Wt = self.link.transform.W.data
        Wt[...] = numpy.random.uniform(-1, 1, Wt.shape)
        bt = self.link.transform.b.data
        bt[...] = numpy.random.uniform(-1, 1, bt.shape)
        self.link.cleargrads()
        self.Wh = Wh.copy()
        self.bh = bh.copy()
        self.Wt = Wt.copy()
        self.bt = bt.copy()
        a = numpy.tanh(self.x.dot(Wh.T) + bh)
        b = self.sigmoid(self.x.dot(Wt.T) + bt)
        self.y = a * b + self.x * (numpy.ones_like(self.x) - b)

    def sigmoid(self, x):
        if False:
            for i in range(10):
                print('nop')
        half = x.dtype.type(0.5)
        return numpy.tanh(x * half) * half + half

    def check_forward(self, x_data):
        if False:
            while True:
                i = 10
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        testing.assert_allclose(self.y, y.data)

    def test_forward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        if False:
            return 10
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        if False:
            print('Hello World!')
        gradient_check.check_backward(self.link, x_data, y_grad, (self.link.plain.W, self.link.plain.b, self.link.transform.W, self.link.transform.b), eps=0.01, atol=0.0032, rtol=0.01)

    def test_backward_cpu(self):
        if False:
            print('Hello World!')
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))
testing.run_module(__name__, __file__)