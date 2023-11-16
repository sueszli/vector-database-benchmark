import unittest
import numpy as np
import parameterize as param
import scipy.stats
from distribution import config
from distribution.config import ATOL, RTOL
from parameterize import xrand
import paddle
np.random.seed(2022)
paddle.enable_static()

@param.place(config.DEVICES)
@param.parameterize_cls((param.TEST_CASE_NAME, 'alpha', 'beta'), [('test-tensor', xrand((10, 10)), xrand((10, 10))), ('test-broadcast', xrand((2, 1)), xrand((2, 5))), ('test-larger-data', xrand((10, 20)), xrand((10, 20)))])
class TestBeta(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            alpha = paddle.static.data('alpha', self.alpha.shape, self.alpha.dtype)
            beta = paddle.static.data('beta', self.beta.shape, self.beta.dtype)
            self._paddle_beta = paddle.distribution.Beta(alpha, beta)
            self.feeds = {'alpha': self.alpha, 'beta': self.beta}

    def test_mean(self):
        if False:
            for i in range(10):
                print('nop')
        with paddle.static.program_guard(self.program):
            [mean] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_beta.mean])
            np.testing.assert_allclose(mean, scipy.stats.beta.mean(self.alpha, self.beta), rtol=RTOL.get(str(self.alpha.dtype)), atol=ATOL.get(str(self.alpha.dtype)))

    def test_variance(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(self.program):
            [variance] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_beta.variance])
            np.testing.assert_allclose(variance, scipy.stats.beta.var(self.alpha, self.beta), rtol=RTOL.get(str(self.alpha.dtype)), atol=ATOL.get(str(self.alpha.dtype)))

    def test_prob(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            value = paddle.static.data('value', self._paddle_beta.alpha.shape, self._paddle_beta.alpha.dtype)
            prob = self._paddle_beta.prob(value)
            random_number = np.random.rand(*self._paddle_beta.alpha.shape)
            feeds = dict(self.feeds, value=random_number)
            [prob] = self.executor.run(self.program, feed=feeds, fetch_list=[prob])
            np.testing.assert_allclose(prob, scipy.stats.beta.pdf(random_number, self.alpha, self.beta), rtol=RTOL.get(str(self.alpha.dtype)), atol=ATOL.get(str(self.alpha.dtype)))

    def test_log_prob(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(self.program):
            value = paddle.static.data('value', self._paddle_beta.alpha.shape, self._paddle_beta.alpha.dtype)
            prob = self._paddle_beta.log_prob(value)
            random_number = np.random.rand(*self._paddle_beta.alpha.shape)
            feeds = dict(self.feeds, value=random_number)
            [prob] = self.executor.run(self.program, feed=feeds, fetch_list=[prob])
            np.testing.assert_allclose(prob, scipy.stats.beta.logpdf(random_number, self.alpha, self.beta), rtol=RTOL.get(str(self.alpha.dtype)), atol=ATOL.get(str(self.alpha.dtype)))

    def test_entropy(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(self.program):
            [entropy] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_beta.entropy()])
            np.testing.assert_allclose(entropy, scipy.stats.beta.entropy(self.alpha, self.beta), rtol=RTOL.get(str(self.alpha.dtype)), atol=ATOL.get(str(self.alpha.dtype)))

    def test_sample(self):
        if False:
            i = 10
            return i + 15
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(self.program, feed=self.feeds, fetch_list=self._paddle_beta.sample())
            self.assertTrue(data.shape, np.broadcast_arrays(self.alpha, self.beta)[0].shape)