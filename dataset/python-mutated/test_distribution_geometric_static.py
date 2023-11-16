import unittest
import numpy as np
import scipy.stats
from distribution.config import ATOL, DEVICES, RTOL
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand
import paddle
from paddle.distribution import geometric
np.random.seed(2023)
paddle.enable_static()

@place(DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'probs'), [('one-dim', xrand((2,), dtype='float32', min=np.finfo(dtype='float32').tiny, max=1.0)), ('multi-dim', xrand((2, 3), dtype='float32', min=np.finfo(dtype='float32').tiny, max=1.0))])
class TestGeometric(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            probs = paddle.static.data('probs', self.probs.shape, self.probs.dtype)
            self._paddle_geometric = geometric.Geometric(probs)
            self.feeds = {'probs': self.probs}

    def test_mean(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            [mean] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_geometric.mean])
            np.testing.assert_allclose(mean, scipy.stats.geom.mean(self.probs, loc=-1), rtol=RTOL.get(str(self.probs.dtype)), atol=ATOL.get(str(self.probs.dtype)))

    def test_variance(self):
        if False:
            return 10
        with paddle.static.program_guard(self.program):
            [variance] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_geometric.variance])
            np.testing.assert_allclose(variance, scipy.stats.geom.var(self.probs, loc=-1), rtol=RTOL.get(str(self.probs.dtype)), atol=ATOL.get(str(self.probs.dtype)))

    def test_stddev(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            [stddev] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_geometric.stddev])
            np.testing.assert_allclose(stddev, scipy.stats.geom.std(self.probs, loc=-1), rtol=RTOL.get(str(self.probs.dtype)), atol=ATOL.get(str(self.probs.dtype)))

    def test_sample(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(self.program, feed=self.feeds, fetch_list=self._paddle_geometric.sample())
            self.assertTrue(data.shape, np.broadcast_arrays(self.probs)[0].shape)

    def test_rsample(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(self.program, feed=self.feeds, fetch_list=self._paddle_geometric.rsample())
            self.assertTrue(data.shape, np.broadcast_arrays(self.probs)[0].shape)

    def test_entropy(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(self.program):
            [entropy] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_geometric.entropy()])
            np.testing.assert_allclose(entropy, scipy.stats.geom.entropy(self.probs, loc=-1), rtol=RTOL.get(str(self.probs.dtype)), atol=ATOL.get(str(self.probs.dtype)))

    def test_init_prob_type_error(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            paddle.distribution.geometric.Geometric([0.5])

@place(DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'probs', 'value'), [('one-dim', xrand((2,), dtype='float32', min=np.finfo(dtype='float32').tiny, max=1.0), 5), ('mult-dim', xrand((2, 2), dtype='float32', min=np.finfo(dtype='float32').tiny, max=1.0), 5), ('mult-dim', xrand((2, 2, 2), dtype='float32', min=np.finfo(dtype='float32').tiny, max=1.0), 5)])
class TestGeometricPMF(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            probs = paddle.static.data('probs', self.probs.shape, self.probs.dtype)
            self._paddle_geometric = geometric.Geometric(probs)
            self.feeds = {'probs': self.probs, 'value': self.value}

    def test_pmf(self):
        if False:
            i = 10
            return i + 15
        with paddle.static.program_guard(self.program):
            [pmf] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_geometric.pmf(self.value)])
            np.testing.assert_allclose(pmf, scipy.stats.geom.pmf(self.value, self.probs, loc=-1), rtol=RTOL.get(str(self.probs.dtype)), atol=ATOL.get(str(self.probs.dtype)))

    def test_log_pmf(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            [log_pmf] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_geometric.log_pmf(self.value)])
            np.testing.assert_allclose(log_pmf, scipy.stats.geom.logpmf(self.value, self.probs, loc=-1), rtol=RTOL.get(str(self.probs.dtype)), atol=ATOL.get(str(self.probs.dtype)))

    def test_cdf(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            [cdf] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_geometric.cdf(self.value)])
            np.testing.assert_allclose(cdf, scipy.stats.geom.cdf(self.value, self.probs, loc=-1), rtol=RTOL.get(str(self.probs.dtype)), atol=ATOL.get(str(self.probs.dtype)))

    def test_pmf_error(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, self._paddle_geometric.pmf, [1, 2])

    def test_log_pmf_error(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, self._paddle_geometric.log_pmf, [1, 2])

    def test_cdf_error(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, self._paddle_geometric.cdf, [1, 2])

@place(DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'probs1', 'probs2'), [('one-dim', xrand((2,), dtype='float32', min=np.finfo(dtype='float32').tiny, max=1.0), xrand((2,), dtype='float32', min=np.finfo(dtype='float32').tiny, max=1.0)), ('multi-dim', xrand((2, 2), dtype='float32', min=np.finfo(dtype='float32').tiny, max=1.0), xrand((2, 2), dtype='float32', min=np.finfo(dtype='float32').tiny, max=1.0))])
class TestGeometricKL(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        self.program_p = paddle.static.Program()
        self.program_q = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program_p, self.program_q):
            probs_p = paddle.static.data('probs1', self.probs1.shape, self.probs1.dtype)
            probs_q = paddle.static.data('probs2', self.probs2.shape, self.probs2.dtype)
            self._paddle_geomP = geometric.Geometric(probs_p)
            self._paddle_geomQ = geometric.Geometric(probs_q)
            self.feeds = {'probs1': self.probs1, 'probs2': self.probs2}

    def test_kl_divergence(self):
        if False:
            return 10
        with paddle.static.program_guard(self.program_p, self.program_q):
            self.executor.run(self.program_q)
            [kl_diver] = self.executor.run(self.program_p, feed=self.feeds, fetch_list=[self._paddle_geomP.kl_divergence(self._paddle_geomQ)])
            np.testing.assert_allclose(kl_diver, self._kl(), rtol=RTOL.get(str(self.probs1.dtype)), atol=ATOL.get(str(self.probs1.dtype)))

    def test_kl1_error(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, self._paddle_geomP.kl_divergence, paddle.distribution.beta.Beta)

    def test_kl2_error(self):
        if False:
            return 10
        self.assertRaises(TypeError, self._paddle_geomQ.kl_divergence, paddle.distribution.beta.Beta)

    def _kl(self):
        if False:
            while True:
                i = 10
        return self.probs1 * np.log(self.probs1 / self.probs2) + (1.0 - self.probs1) * np.log((1.0 - self.probs1) / (1.0 - self.probs2))
if __name__ == '__main__':
    unittest.main()