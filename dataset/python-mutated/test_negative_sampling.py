import unittest
import numpy
import chainer
from chainer import backend
from chainer.backend import CpuDevice
from chainer import links
from chainer import testing

@testing.parameterize(*testing.product({'dtype': [numpy.float16, numpy.float32, numpy.float64], 't': [[0, 2], [-1, 1, 2]], 'reduce': ['sum', 'no']}))
@testing.inject_backend_tests(None, [{}, {'use_cuda': True, 'cuda_device': 0}, {'use_cuda': True, 'cuda_device': 1}, {'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestNegativeSampling(unittest.TestCase):
    in_size = 3
    sample_size = 2

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()
        batch = len(self.t)
        x_shape = (batch, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.t = numpy.array(self.t).astype(numpy.int32)
        if self.reduce == 'no':
            g_shape = self.t.shape
        elif self.reduce == 'sum':
            g_shape = ()
        self.gy = numpy.random.uniform(-1, 1, g_shape).astype(self.dtype)
        if self.dtype == numpy.float16:
            self.test_forward_options = {'atol': 0.01}
            self.test_backward_options = {'atol': 0.005}
        else:
            self.test_forward_options = {}
            self.test_backward_options = {'atol': 0.0001}

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self._config_user.__exit__(None, None, None)

    def create_link(self, rng=None):
        if False:
            print('Hello World!')
        if rng is None:
            rng = numpy.random.RandomState()
        link = links.NegativeSampling(self.in_size, [10, 5, 2, 5, 2], self.sample_size)
        link.cleargrads()
        link.W.array[:] = rng.uniform(-1, 1, link.W.shape)
        return link

    def call_link_with_samples(self, samples, func):
        if False:
            i = 10
            return i + 15

        def mock_sample(shape):
            if False:
                i = 10
                return i + 15
            assert samples.shape == shape
            return samples.copy()
        orig_negative_sampling = chainer.functions.negative_sampling

        def wrap_negative_sampling(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            args = args[:3] + (mock_sample,) + args[4:]
            return orig_negative_sampling(*args, **kwargs)
        with testing.patch('chainer.functions.loss.negative_sampling.negative_sampling', wraps=wrap_negative_sampling) as m:
            ret = func()
            assert m.call_count == 1
        return ret

    def test_forward(self, backend_config):
        if False:
            i = 10
            return i + 15
        x_data = backend_config.get_array(self.x)
        t_data = backend_config.get_array(self.t)
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data, requires_grad=False)
        link = self.create_link()
        link.to_device(backend_config.device)
        (y, samples) = link(x, t, reduce=self.reduce, return_samples=True)
        self.assertEqual(y.shape, self.gy.shape)
        cpu_device = CpuDevice()
        W = cpu_device.send(link.W.data)
        samples = cpu_device.send(samples)
        loss = numpy.empty((len(self.x),), self.dtype)
        for i in range(len(self.x)):
            ix = self.x[i]
            it = self.t[i]
            if it == -1:
                loss[i] = 0
            else:
                w = W[samples[i]]
                f = w.dot(ix)
                f[0] *= -1
                loss[i] = numpy.logaddexp(f, 0).sum()
        if self.reduce == 'sum':
            loss = loss.sum()
        testing.assert_allclose(y.data, loss, **self.test_forward_options)

    def test_to_cpu(self, backend_config):
        if False:
            while True:
                i = 10
        link = self.create_link()
        link.to_device(backend_config.device)
        self.assertEqual(link.sampler.device, backend_config.device)
        with testing.assert_warns(DeprecationWarning):
            link.to_cpu()
        self.assertEqual(link.sampler.device, backend.CpuDevice())

    def test_return_samples(self, backend_config):
        if False:
            while True:
                i = 10
        batch_size = self.t.shape[0]
        link = self.create_link()
        link.to_device(backend_config.device)
        x_data = backend_config.get_array(self.x)
        t_data = backend_config.get_array(self.t)
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data, requires_grad=False)
        (y, samples) = link(x, t, reduce=self.reduce, return_samples=True)
        assert isinstance(samples, backend_config.xp.ndarray)
        assert samples.shape == (batch_size, self.sample_size + 1)
        assert samples.dtype == numpy.int32
        y_ = self.call_link_with_samples(samples, lambda : link(x, t, reduce=self.reduce))
        cpu_device = CpuDevice()
        numpy.testing.assert_array_equal(cpu_device.send(y.array), cpu_device.send(y_.array))

    def test_backward_compare_with_numpy(self, backend_config):
        if False:
            print('Hello World!')
        rng = numpy.random.RandomState()
        rng_state = rng.get_state()
        x = chainer.Variable(self.x)
        t = chainer.Variable(self.t, requires_grad=False)
        link = self.create_link(rng)
        (y, samples) = link(x, t, return_samples=True)
        y.backward()
        assert t.grad is None
        gw_cpu = link.W.grad
        gx_cpu = x.grad
        rng.set_state(rng_state)
        link = self.create_link(rng)
        link.to_device(backend_config.device)
        x = chainer.Variable(backend_config.get_array(self.x))
        t = chainer.Variable(backend_config.get_array(self.t), requires_grad=False)
        samples = backend_config.get_array(samples)
        y = self.call_link_with_samples(samples, lambda : link(x, t))
        y.backward()
        assert t.grad is None
        gw_gpu = link.W.grad
        gx_gpu = x.grad
        testing.assert_allclose(gx_cpu, gx_gpu, **self.test_backward_options)
        testing.assert_allclose(gw_cpu, gw_gpu, **self.test_backward_options)
testing.run_module(__name__, __file__)