import unittest
import numpy
import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64]}))
class TestBlackOut(unittest.TestCase):
    batch_size = 5
    in_size = 4
    count = [3, 2, 1]
    n_samples = 7

    def setUp(self):
        if False:
            print('Hello World!')
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()
        x_shape = (self.batch_size, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.t = numpy.random.randint(len(self.count), size=self.batch_size).astype(numpy.int32)
        self.link = links.BlackOut(self.in_size, self.count, self.n_samples)
        self.w = numpy.random.uniform(-1, 1, self.link.W.data.shape)
        self.link.W.data[:] = self.w
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 0.005}
        else:
            self.check_forward_options = {'atol': 0.0001}

    def tearDown(self):
        if False:
            print('Hello World!')
        self._config_user.__exit__(None, None, None)

    def check_forward(self, x_data, t_data):
        if False:
            for i in range(10):
                print('nop')
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data, requires_grad=False)
        self.link.sample_data = self.link.sampler.sample((self.batch_size, self.n_samples))
        y = self.link(x, t)
        expect_y = numpy.empty(self.batch_size, dtype=self.dtype)
        samples = cuda.to_cpu(self.link.sample_data)
        for b in range(self.batch_size):
            z = 0
            for i in range(self.n_samples):
                w = samples[b, i]
                z += numpy.exp(self.w[w].dot(self.x[b]))
            y0 = self.w[self.t[b]].dot(self.x[b])
            z += numpy.exp(y0)
            l = y0 - numpy.log(z)
            for i in range(self.n_samples):
                w = samples[b, i]
                l += numpy.log(1 - numpy.exp(self.w[w].dot(self.x[b])) / z)
            expect_y[b] = l
        loss = -numpy.sum(expect_y) / self.batch_size
        testing.assert_allclose(y.data, loss, **self.check_forward_options)

    def test_forward_cpu(self):
        if False:
            print('Hello World!')
        self.check_forward(self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        if False:
            i = 10
            return i + 15
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.chainerx
    def test_forward_chainerx_native(self):
        if False:
            while True:
                i = 10
        device = chainer.get_device('native:0')
        self.link.to_device(device)
        self.check_forward(device.send(self.x), device.send(self.t))

    @attr.chainerx
    @attr.gpu
    def test_forward_chainerx_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        device = chainer.get_device('cuda:0')
        self.link.to_device(device)
        self.check_forward(device.send(self.x), device.send(self.t))
testing.run_module(__name__, __file__)