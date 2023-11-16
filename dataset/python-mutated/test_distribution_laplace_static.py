import unittest
import numpy as np
import parameterize
import scipy.stats
from distribution import config
import paddle
paddle.enable_static()

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'loc', 'scale'), [('one-dim', parameterize.xrand((2,)), parameterize.xrand((2,))), ('multi-dim', parameterize.xrand((5, 5)), parameterize.xrand((5, 5)))])
class TestLaplace(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data('scale', self.scale.shape, self.scale.dtype)
            self._dist = paddle.distribution.Laplace(loc=loc, scale=scale)
            self.sample_shape = (50000,)
            mean = self._dist.mean
            var = self._dist.variance
            stddev = self._dist.stddev
            entropy = self._dist.entropy()
            samples = self._dist.sample(self.sample_shape)
        fetch_list = [mean, var, stddev, entropy, samples]
        self.feeds = {'loc': self.loc, 'scale': self.scale}
        executor.run(startup_program)
        [self.mean, self.var, self.stddev, self.entropy, self.samples] = executor.run(main_program, feed=self.feeds, fetch_list=fetch_list)

    def test_mean(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(str(self.mean.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(self.mean, self._np_mean(), rtol=config.RTOL.get(str(self.scale.dtype)), atol=config.ATOL.get(str(self.scale.dtype)))

    def test_variance(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(str(self.var.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(self.var, self._np_variance(), rtol=config.RTOL.get(str(self.scale.dtype)), atol=config.ATOL.get(str(self.scale.dtype)))

    def test_stddev(self):
        if False:
            print('Hello World!')
        self.assertEqual(str(self.stddev.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(self.stddev, self._np_stddev(), rtol=config.RTOL.get(str(self.scale.dtype)), atol=config.ATOL.get(str(self.scale.dtype)))

    def test_entropy(self):
        if False:
            print('Hello World!')
        self.assertEqual(str(self.entropy.dtype).split('.')[-1], self.scale.dtype)

    def test_sample(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.samples.dtype, self.scale.dtype)
        self.assertEqual(tuple(self.samples.shape), tuple(self._dist._extend_shape(self.sample_shape)))
        self.assertEqual(self.samples.shape, self.sample_shape + self.loc.shape)
        self.assertEqual(self.samples.shape, self.sample_shape + self.loc.shape)
        np.testing.assert_allclose(self.samples.mean(axis=0), scipy.stats.laplace.mean(self.loc, scale=self.scale), rtol=0.2, atol=0.0)
        np.testing.assert_allclose(self.samples.var(axis=0), scipy.stats.laplace.var(self.loc, scale=self.scale), rtol=0.1, atol=0.0)

    def _np_mean(self):
        if False:
            while True:
                i = 10
        return self.loc

    def _np_stddev(self):
        if False:
            for i in range(10):
                print('nop')
        return 2 ** 0.5 * self.scale

    def _np_variance(self):
        if False:
            for i in range(10):
                print('nop')
        stddev = 2 ** 0.5 * self.scale
        return np.power(stddev, 2)

    def _np_entropy(self):
        if False:
            i = 10
            return i + 15
        return scipy.stats.laplace.entropy(loc=self.loc, scale=self.scale)

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'loc', 'scale', 'value'), [('value-float', np.array([0.2, 0.3]), np.array([2.0, 3.0]), np.array([2.0, 5.0])), ('value-int', np.array([0.2, 0.3]), np.array([2.0, 3.0]), np.array([2, 5])), ('value-multi-dim', np.array([0.2, 0.3]), np.array([2.0, 3.0]), np.array([[4.0, 6], [8, 2]]))])
class TestLaplacePDF(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data('scale', self.scale.shape, self.scale.dtype)
            value = paddle.static.data('value', self.value.shape, self.value.dtype)
            self._dist = paddle.distribution.Laplace(loc=loc, scale=scale)
            prob = self._dist.prob(value)
            log_prob = self._dist.log_prob(value)
            cdf = self._dist.cdf(value)
            icdf = self._dist.icdf(value)
        fetch_list = [prob, log_prob, cdf, icdf]
        self.feeds = {'loc': self.loc, 'scale': self.scale, 'value': self.value}
        executor.run(startup_program)
        [self.prob, self.log_prob, self.cdf, self.icdf] = executor.run(main_program, feed=self.feeds, fetch_list=fetch_list)

    def test_prob(self):
        if False:
            return 10
        np.testing.assert_allclose(self.prob, scipy.stats.laplace.pdf(self.value, self.loc, self.scale), rtol=config.RTOL.get(str(self.loc.dtype)), atol=config.ATOL.get(str(self.loc.dtype)))

    def test_log_prob(self):
        if False:
            print('Hello World!')
        np.testing.assert_allclose(self.log_prob, scipy.stats.laplace.logpdf(self.value, self.loc, self.scale), rtol=config.RTOL.get(str(self.loc.dtype)), atol=config.ATOL.get(str(self.loc.dtype)))

    def test_cdf(self):
        if False:
            i = 10
            return i + 15
        np.testing.assert_allclose(self.cdf, scipy.stats.laplace.cdf(self.value, self.loc, self.scale), rtol=config.RTOL.get(str(self.loc.dtype)), atol=config.ATOL.get(str(self.loc.dtype)))

    def test_icdf(self):
        if False:
            return 10
        np.testing.assert_allclose(self.icdf, scipy.stats.laplace.ppf(self.value, self.loc, self.scale), rtol=config.RTOL.get(str(self.loc.dtype)), atol=config.ATOL.get(str(self.loc.dtype)))

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'loc1', 'scale1', 'loc2', 'scale2'), [('kl', np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([0.5]))])
class TestLaplaceAndLaplaceKL(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.mp = paddle.static.Program()
        self.sp = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.mp, self.sp):
            loc1 = paddle.static.data('loc1', self.loc1.shape, self.loc1.dtype)
            scale1 = paddle.static.data('scale1', self.scale1.shape, self.scale1.dtype)
            loc2 = paddle.static.data('loc2', self.loc2.shape, self.loc2.dtype)
            scale2 = paddle.static.data('scale2', self.scale2.shape, self.scale2.dtype)
            self._dist_1 = paddle.distribution.Laplace(loc=loc1, scale=scale1)
            self._dist_2 = paddle.distribution.Laplace(loc=loc2, scale=scale2)
            self.feeds = {'loc1': self.loc1, 'scale1': self.scale1, 'loc2': self.loc2, 'scale2': self.scale2}

    def test_kl_divergence(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(self.mp, self.sp):
            out = paddle.distribution.kl_divergence(self._dist_1, self._dist_2)
            self.executor.run(self.sp)
            [out] = self.executor.run(self.mp, feed=self.feeds, fetch_list=[out])
            np.testing.assert_allclose(out, self._np_kl(), atol=0, rtol=0.5)

    def _np_kl(self):
        if False:
            while True:
                i = 10
        x = np.linspace(scipy.stats.laplace.ppf(0.01), scipy.stats.laplace.ppf(0.99), 1000)
        d1 = scipy.stats.laplace.pdf(x, loc=0.0, scale=1.0)
        d2 = scipy.stats.laplace.pdf(x, loc=1.0, scale=0.5)
        return scipy.stats.entropy(d1, d2)
"\n# Note: Zero dimension of a Tensor is not supported by static graph mode of paddle;\n# therefore, ks test below cannot be conducted temporarily.\n\n@parameterize.place(config.DEVICES)\n@parameterize.parameterize_cls(\n    (parameterize.TEST_CASE_NAME, 'loc', 'scale', 'sample_shape'), [\n    ('one-dim', np.array(4.0), np.array(3.0), np.array([3000]))])\nclass TestLaplaceKS(unittest.TestCase):\n    def setUp(self):\n        self.program = paddle.static.Program()\n        self.executor = paddle.static.Executor(self.place)\n        with paddle.static.program_guard(self.program):\n            loc = paddle.static.data('loc', self.loc.shape,\n                                        self.loc.dtype)\n            scale = paddle.static.data('scale', self.scale.shape,\n                                        self.scale.dtype)\n            self.sample = paddle.static.data('sample_shape', self.sample_shape.shape,\n                                        self.sample_shape.dtype)\n            self._dist = paddle.distribution.Laplace(loc=loc, scale=scale)\n            self.feeds = {'loc': self.loc, 'scale': self.scale, 'sample_shape': self.sample_shape}\n\n    def test_sample(self):\n        with paddle.static.program_guard(self.program):\n            [sample_values] = self.executor.run(self.program,\n                              feed=self.feeds,\n                              fetch_list=self._dist.sample((3000,)))\n            self.assertTrue(self._kstest(self.loc, self.scale, sample_values))\n\n    def _kstest(self, loc, scale, samples):\n        # Uses the Kolmogorov-Smirnov test for goodness of fit.\n        ks, p_value = scipy.stats.kstest(\n            samples,\n            scipy.stats.laplace(loc, scale=scale).cdf)\n        return ks < 0.02\n"
if __name__ == '__main__':
    unittest.main()