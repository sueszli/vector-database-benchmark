import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

@testing.parameterize({'x_data': [0, 1, 0], 'ignore_label': None}, {'x_data': [[0, 1, 0], [1, 0, 1]], 'ignore_label': None}, {'x_data': [0, 1, -1], 'ignore_label': -1}, {'x_data': [[0, 1, -1], [-1, 0, 1]], 'ignore_label': -1})
class TestEmbedID(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.link = links.EmbedID(3, 2, ignore_label=self.ignore_label)
        self.link.ignore_label
        self.link.cleargrads()
        self.W = self.link.W.data.copy()
        self.x = numpy.array(self.x_data, dtype=numpy.int32)
        y_shape = self.x.shape + (2,)
        self.gy = numpy.random.uniform(-1, 1, y_shape).astype(numpy.float32)

    def check_forward(self, x_data):
        if False:
            while True:
                i = 10
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_expect = numpy.empty_like(self.gy)
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] == -1:
                y_expect[i] = 0
            else:
                y_expect[i] = self.W[int(self.x[i])]
        testing.assert_allclose(y_expect, y.data, atol=0, rtol=0)

    @condition.retry(3)
    def test_forward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        if False:
            i = 10
            return i + 15
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    def test_forward_mixed_cpu_gpu_1(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    def test_forward_mixed_cpu_gpu_2(self):
        if False:
            for i in range(10):
                print('nop')
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        with self.assertRaises(TypeError):
            self.check_forward(self.x)

    def check_backward(self, x_data, y_grad):
        if False:
            return 10
        gradient_check.check_backward(self.link, x_data, y_grad, self.link.W, atol=0.0001, rtol=0.001)

    @condition.retry(3)
    def test_backward_cpu(self):
        if False:
            while True:
                i = 10
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        if False:
            i = 10
            return i + 15
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

@testing.parameterize({'t_value': -1, 'valid': False, 'ignore_label': None}, {'t_value': 3, 'valid': False, 'ignore_label': None}, {'t_value': 0, 'valid': True, 'ignore_label': None}, {'t_value': -1, 'valid': True, 'ignore_label': -1}, {'t_value': 3, 'valid': False, 'ignore_label': -1}, {'t_value': 0, 'valid': True, 'ignore_label': -1})
class TestEmbedIDValueCheck(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.link = links.EmbedID(2, 2, ignore_label=self.ignore_label)
        self.t = numpy.array([self.t_value], dtype=numpy.int32)
        self.original_debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        if False:
            return 10
        chainer.set_debug(self.original_debug)

    def check_value_check(self, t_data):
        if False:
            return 10
        t = chainer.Variable(t_data)
        if self.valid:
            self.link(t)
        else:
            with self.assertRaises(ValueError):
                self.link(t)

    def test_value_check_cpu(self):
        if False:
            return 10
        self.check_value_check(self.t)

    @attr.gpu
    def test_value_check_gpu(self):
        if False:
            while True:
                i = 10
        self.check_value_check(self.t)

class TestEmbedIDUnpickleOldFile(unittest.TestCase):

    def test_old_unpickle(self):
        if False:
            print('Hello World!')
        embed = links.EmbedID(3, 4)
        delattr(embed, 'ignore_label')
        x = chainer.Variable(numpy.arange(2, dtype=numpy.int32))
        y = embed(x)
        self.assertEqual(y.data.shape, (2, 4))

class TestEmbedIDFromParams(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        (self.in_size, self.out_size) = (10, 5)

    def test_from_params(self):
        if False:
            print('Hello World!')
        link1 = links.EmbedID(self.in_size, self.out_size)
        link2 = links.EmbedID.from_params(link1.W)
        assert link2.W.shape == link1.W.shape
testing.run_module(__name__, __file__)