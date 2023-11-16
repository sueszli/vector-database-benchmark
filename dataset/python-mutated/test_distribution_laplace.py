import unittest
import numpy as np
import parameterize
import scipy.stats
from distribution import config
import paddle

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'loc', 'scale'), [('one-dim', parameterize.xrand((2,)), parameterize.xrand((2,))), ('multi-dim', parameterize.xrand((5, 5)), parameterize.xrand((5, 5)))])
class TestLaplace(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._dist = paddle.distribution.Laplace(loc=paddle.to_tensor(self.loc), scale=paddle.to_tensor(self.scale))

    def test_mean(self):
        if False:
            for i in range(10):
                print('nop')
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.scale.dtype)
        np.testing.assert_allclose(mean, self._np_mean(), rtol=config.RTOL.get(str(self.scale.dtype)), atol=config.ATOL.get(str(self.scale.dtype)))

    def test_variance(self):
        if False:
            print('Hello World!')
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.scale.dtype)
        np.testing.assert_allclose(var, self._np_variance(), rtol=config.RTOL.get(str(self.scale.dtype)), atol=config.ATOL.get(str(self.scale.dtype)))

    def test_stddev(self):
        if False:
            print('Hello World!')
        stddev = self._dist.stddev
        self.assertEqual(stddev.numpy().dtype, self.scale.dtype)
        np.testing.assert_allclose(stddev, self._np_stddev(), rtol=config.RTOL.get(str(self.scale.dtype)), atol=config.ATOL.get(str(self.scale.dtype)))

    def test_entropy(self):
        if False:
            return 10
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.scale.dtype)

    def test_sample(self):
        if False:
            return 10
        sample_shape = (50000,)
        samples = self._dist.sample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(samples.numpy().dtype, self.scale.dtype)
        self.assertEqual(tuple(samples.shape), tuple(self._dist._extend_shape(sample_shape)))
        self.assertEqual(samples.shape, list(sample_shape + self.loc.shape))
        self.assertEqual(sample_values.shape, sample_shape + self.loc.shape)
        np.testing.assert_allclose(sample_values.mean(axis=0), scipy.stats.laplace.mean(self.loc, scale=self.scale), rtol=0.2, atol=0.0)
        np.testing.assert_allclose(sample_values.var(axis=0), scipy.stats.laplace.var(self.loc, scale=self.scale), rtol=0.1, atol=0.0)

    def _np_mean(self):
        if False:
            print('Hello World!')
        return self.loc

    def _np_stddev(self):
        if False:
            print('Hello World!')
        return 2 ** 0.5 * self.scale

    def _np_variance(self):
        if False:
            for i in range(10):
                print('nop')
        stddev = 2 ** 0.5 * self.scale
        return np.power(stddev, 2)

    def _np_entropy(self):
        if False:
            for i in range(10):
                print('nop')
        return scipy.stats.laplace.entropy(loc=self.loc, scale=self.scale)

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'loc', 'scale'), [('float', 1.0, 2.0), ('int', 3, 4)])
class TestLaplaceKS(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._dist = paddle.distribution.Laplace(loc=self.loc, scale=self.scale)

    def test_sample(self):
        if False:
            print('Hello World!')
        sample_shape = (20000,)
        samples = self._dist.sample(sample_shape)
        sample_values = samples.numpy()
        self.assertTrue(self._kstest(self.loc, self.scale, sample_values))

    def _kstest(self, loc, scale, samples):
        if False:
            print('Hello World!')
        (ks, p_value) = scipy.stats.kstest(samples, scipy.stats.laplace(loc, scale=scale).cdf)
        return ks < 0.02

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'loc', 'scale', 'value'), [('value-float', np.array([0.2, 0.3]), np.array([2.0, 3.0]), np.array([2.0, 5.0])), ('value-int', np.array([0.2, 0.3]), np.array([2.0, 3.0]), np.array([2, 5])), ('value-multi-dim', np.array([0.2, 0.3]), np.array([2.0, 3.0]), np.array([[4.0, 6], [8, 2]]))])
class TestLaplacePDF(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._dist = paddle.distribution.Laplace(loc=paddle.to_tensor(self.loc), scale=paddle.to_tensor(self.scale))

    def test_prob(self):
        if False:
            while True:
                i = 10
        np.testing.assert_allclose(self._dist.prob(paddle.to_tensor(self.value)), scipy.stats.laplace.pdf(self.value, self.loc, self.scale), rtol=config.RTOL.get(str(self.loc.dtype)), atol=config.ATOL.get(str(self.loc.dtype)))

    def test_log_prob(self):
        if False:
            return 10
        np.testing.assert_allclose(self._dist.log_prob(paddle.to_tensor(self.value)), scipy.stats.laplace.logpdf(self.value, self.loc, self.scale), rtol=config.RTOL.get(str(self.loc.dtype)), atol=config.ATOL.get(str(self.loc.dtype)))

    def test_cdf(self):
        if False:
            i = 10
            return i + 15
        np.testing.assert_allclose(self._dist.cdf(paddle.to_tensor(self.value)), scipy.stats.laplace.cdf(self.value, self.loc, self.scale), rtol=config.RTOL.get(str(self.loc.dtype)), atol=config.ATOL.get(str(self.loc.dtype)))

    def test_icdf(self):
        if False:
            return 10
        np.testing.assert_allclose(self._dist.icdf(paddle.to_tensor(self.value)), scipy.stats.laplace.ppf(self.value, self.loc, self.scale), rtol=config.RTOL.get(str(self.loc.dtype)), atol=config.ATOL.get(str(self.loc.dtype)))

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'loc1', 'scale1', 'loc2', 'scale2'), [('kl', np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([0.5]))])
class TestLaplaceAndLaplaceKL(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._dist_1 = paddle.distribution.Laplace(loc=paddle.to_tensor(self.loc1), scale=paddle.to_tensor(self.scale1))
        self._dist_2 = paddle.distribution.Laplace(loc=paddle.to_tensor(self.loc2), scale=paddle.to_tensor(self.scale2))

    def test_kl_divergence(self):
        if False:
            print('Hello World!')
        np.testing.assert_allclose(paddle.distribution.kl_divergence(self._dist_1, self._dist_2), self._np_kl(), atol=0, rtol=0.5)

    def _np_kl(self):
        if False:
            while True:
                i = 10
        x = np.linspace(scipy.stats.laplace.ppf(0.01), scipy.stats.laplace.ppf(0.99), 1000)
        d1 = scipy.stats.laplace.pdf(x, loc=0.0, scale=1.0)
        d2 = scipy.stats.laplace.pdf(x, loc=1.0, scale=0.5)
        return scipy.stats.entropy(d1, d2)
if __name__ == '__main__':
    unittest.main()