import unittest
import numpy as np
import parameterize
import scipy.stats
from distribution import config
import paddle

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'total_count', 'probs'), [('one-dim', 10, parameterize.xrand((3,))), ('multi-dim', 9, parameterize.xrand((10, 20))), ('prob-sum-one', 10, np.array([0.5, 0.2, 0.3])), ('prob-sum-non-one', 10, np.array([2.0, 3.0, 5.0]))])
class TestMultinomial(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._dist = paddle.distribution.Multinomial(total_count=self.total_count, probs=paddle.to_tensor(self.probs))

    def test_mean(self):
        if False:
            return 10
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.probs.dtype)
        np.testing.assert_allclose(mean, self._np_mean(), rtol=config.RTOL.get(str(self.probs.dtype)), atol=config.ATOL.get(str(self.probs.dtype)))

    def test_variance(self):
        if False:
            return 10
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.probs.dtype)
        np.testing.assert_allclose(var, self._np_variance(), rtol=config.RTOL.get(str(self.probs.dtype)), atol=config.ATOL.get(str(self.probs.dtype)))

    def test_entropy(self):
        if False:
            for i in range(10):
                print('nop')
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.probs.dtype)
        np.testing.assert_allclose(entropy, self._np_entropy(), rtol=config.RTOL.get(str(self.probs.dtype)), atol=config.ATOL.get(str(self.probs.dtype)))

    def test_sample(self):
        if False:
            return 10
        sample_shape = ()
        samples = self._dist.sample(sample_shape)
        self.assertEqual(samples.numpy().dtype, self.probs.dtype)
        self.assertEqual(tuple(samples.shape), sample_shape + self._dist.batch_shape + self._dist.event_shape)
        sample_shape = (6,)
        samples = self._dist.sample(sample_shape)
        self.assertEqual(samples.numpy().dtype, self.probs.dtype)
        self.assertEqual(tuple(samples.shape), sample_shape + self._dist.batch_shape + self._dist.event_shape)
        self.assertTrue(np.all(samples.sum(-1).numpy() == self._dist.total_count))
        sample_shape = (5000,)
        samples = self._dist.sample(sample_shape)
        sample_mean = samples.mean(axis=0)
        np.testing.assert_allclose(sample_mean, self._dist.mean, atol=0, rtol=0.2)

    def _np_variance(self):
        if False:
            for i in range(10):
                print('nop')
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return self.total_count * probs * (1 - probs)

    def _np_mean(self):
        if False:
            while True:
                i = 10
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return self.total_count * probs

    def _np_entropy(self):
        if False:
            for i in range(10):
                print('nop')
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return scipy.stats.multinomial.entropy(self.total_count, probs)

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'total_count', 'probs', 'value'), [('value-float', 10, np.array([0.2, 0.3, 0.5]), np.array([2.0, 3.0, 5.0])), ('value-int', 10, np.array([0.2, 0.3, 0.5]), np.array([2, 3, 5])), ('value-multi-dim', 10, np.array([[0.3, 0.7], [0.5, 0.5]]), np.array([[4.0, 6], [8, 2]]))])
class TestMultinomialPmf(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._dist = paddle.distribution.Multinomial(total_count=self.total_count, probs=paddle.to_tensor(self.probs))

    def test_prob(self):
        if False:
            for i in range(10):
                print('nop')
        np.testing.assert_allclose(self._dist.prob(paddle.to_tensor(self.value)), scipy.stats.multinomial.pmf(self.value, self.total_count, self.probs), rtol=config.RTOL.get(str(self.probs.dtype)), atol=config.ATOL.get(str(self.probs.dtype)))

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((config.TEST_CASE_NAME, 'total_count', 'probs'), [('total_count_le_one', 0, np.array([0.3, 0.7])), ('total_count_float', np.array([0.3, 0.7])), ('probs_zero_dim', np.array(0))])
class TestMultinomialException(unittest.TestCase):

    def TestInit(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            paddle.distribution.Multinomial(self.total_count, paddle.to_tensor(self.probs))
if __name__ == '__main__':
    unittest.main()