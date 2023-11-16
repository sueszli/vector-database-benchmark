import unittest
import numpy
import six
import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64], 'batchsize': [5, 10], 'input_dim': [2, 3], 'margin': [0.1, 0.5], 'reduce': ['mean', 'no']}))
class TestTriplet(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if self.dtype == numpy.float16:
            eps = 0.01
            self.check_forward_options = {'rtol': 0.005, 'atol': 0.005}
            self.check_backward_options = {'eps': eps, 'rtol': 0.05, 'atol': 0.05}
            self.check_double_backward_options = {'eps': eps, 'rtol': 0.05, 'atol': 0.05}
        elif self.dtype == numpy.float32:
            eps = 0.001
            self.check_forward_options = {'rtol': 0.0001, 'atol': 0.0001}
            self.check_backward_options = {'eps': eps, 'rtol': 0.0005, 'atol': 0.0005}
            self.check_double_backward_options = {'eps': eps, 'rtol': 0.001, 'atol': 0.001}
        elif self.dtype == numpy.float64:
            eps = 0.001
            self.check_forward_options = {'rtol': 0.0001, 'atol': 0.0001}
            self.check_backward_options = {'eps': eps, 'rtol': 0.0005, 'atol': 0.0005}
            self.check_double_backward_options = {'eps': eps, 'rtol': 0.001, 'atol': 0.001}
        else:
            assert False
        x_shape = (self.batchsize, self.input_dim)
        while True:
            self.a = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            self.p = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            self.n = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            if (abs(self.a - self.p) < 2 * eps).any():
                continue
            if (abs(self.a - self.n) < 2 * eps).any():
                continue
            dist = numpy.sum((self.a - self.p) ** 2 - (self.a - self.n) ** 2, axis=1) + self.margin
            if (abs(dist) < 4 * eps).any():
                continue
            break
        if self.reduce == 'mean':
            gy_shape = ()
        else:
            gy_shape = (self.batchsize,)
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)
        self.gga = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.ggp = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.ggn = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

    def check_forward(self, a_data, p_data, n_data):
        if False:
            while True:
                i = 10
        a_val = chainer.Variable(a_data)
        p_val = chainer.Variable(p_data)
        n_val = chainer.Variable(n_data)
        loss = functions.triplet(a_val, p_val, n_val, self.margin, self.reduce)
        if self.reduce == 'mean':
            self.assertEqual(loss.data.shape, ())
        else:
            self.assertEqual(loss.data.shape, (self.batchsize,))
        self.assertEqual(loss.data.dtype, self.dtype)
        loss_value = cuda.to_cpu(loss.data)
        loss_expect = numpy.empty((self.a.shape[0],), dtype=self.dtype)
        for i in six.moves.range(self.a.shape[0]):
            (ad, pd, nd) = (self.a[i], self.p[i], self.n[i])
            dp = numpy.sum((ad - pd) ** 2)
            dn = numpy.sum((ad - nd) ** 2)
            loss_expect[i] = max(dp - dn + self.margin, 0)
        if self.reduce == 'mean':
            loss_expect = loss_expect.mean()
        numpy.testing.assert_allclose(loss_expect, loss_value, **self.check_forward_options)

    def test_negative_margin(self):
        if False:
            while True:
                i = 10
        self.margin = -1
        self.assertRaises(ValueError, self.check_forward, self.a, self.p, self.n)
        self.assertRaises(ValueError, self.check_backward, self.a, self.p, self.n, self.gy)

    def test_forward_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_forward(self.a, self.p, self.n)

    @attr.gpu
    def test_forward_gpu(self):
        if False:
            print('Hello World!')
        self.check_forward(cuda.to_gpu(self.a), cuda.to_gpu(self.p), cuda.to_gpu(self.n))

    def check_backward(self, a_data, p_data, n_data, gy_data):
        if False:
            i = 10
            return i + 15

        def f(a, p, n):
            if False:
                return 10
            return functions.triplet(a, p, n, margin=self.margin, reduce=self.reduce)
        gradient_check.check_backward(f, (a_data, p_data, n_data), gy_data, dtype=numpy.float64, **self.check_backward_options)

    def test_backward_cpu(self):
        if False:
            print('Hello World!')
        self.check_backward(self.a, self.p, self.n, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_backward(cuda.to_gpu(self.a), cuda.to_gpu(self.p), cuda.to_gpu(self.n), cuda.to_gpu(self.gy))

    def check_double_backward(self, a_data, p_data, n_data, gy_data, gga_data, ggp_data, ggn_data):
        if False:
            for i in range(10):
                print('nop')

        def f(a, p, n):
            if False:
                print('Hello World!')
            return functions.triplet(a, p, n, margin=self.margin, reduce=self.reduce)
        gradient_check.check_double_backward(f, (a_data, p_data, n_data), gy_data, (gga_data, ggp_data, ggn_data), dtype=numpy.float64, **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        if False:
            i = 10
            return i + 15
        self.check_double_backward(self.a, self.p, self.n, self.gy, self.gga, self.ggp, self.ggn)

    @attr.gpu
    def test_double_backward_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_double_backward(cuda.to_gpu(self.a), cuda.to_gpu(self.p), cuda.to_gpu(self.n), cuda.to_gpu(self.gy), cuda.to_gpu(self.gga), cuda.to_gpu(self.ggp), cuda.to_gpu(self.ggn))

class TestContrastiveInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.a = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.p = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.n = numpy.random.randint(-1, 1, (5, 10)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        if False:
            print('Hello World!')
        a = xp.asarray(self.a)
        p = xp.asarray(self.p)
        n = xp.asarray(self.n)
        with self.assertRaises(ValueError):
            functions.triplet(a, p, n, reduce='invalid_option')

    def test_invalid_option_cpu(self):
        if False:
            return 10
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        if False:
            i = 10
            return i + 15
        self.check_invalid_option(cuda.cupy)
testing.run_module(__name__, __file__)