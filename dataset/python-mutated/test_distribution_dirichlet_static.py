import unittest
import numpy as np
import scipy.stats
from distribution.config import ATOL, DEVICES, RTOL
from parameterize import TEST_CASE_NAME, parameterize_cls, place
import paddle
np.random.seed(2022)
paddle.enable_static()

@place(DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'concentration'), [('test-one-dim', np.random.rand(89) + 5.0)])
class TestDirichlet(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor()
        with paddle.static.program_guard(self.program):
            conc = paddle.static.data('conc', self.concentration.shape, self.concentration.dtype)
            self._paddle_diric = paddle.distribution.Dirichlet(conc)
            self.feeds = {'conc': self.concentration}

    def test_mean(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            [out] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_diric.mean])
            np.testing.assert_allclose(out, scipy.stats.dirichlet.mean(self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))

    def test_variance(self):
        if False:
            for i in range(10):
                print('nop')
        with paddle.static.program_guard(self.program):
            [out] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_diric.variance])
            np.testing.assert_allclose(out, scipy.stats.dirichlet.var(self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))

    def test_prob(self):
        if False:
            i = 10
            return i + 15
        with paddle.static.program_guard(self.program):
            random_number = np.random.rand(*self.concentration.shape)
            random_number = random_number / random_number.sum()
            feeds = dict(self.feeds, value=random_number)
            value = paddle.static.data('value', random_number.shape, random_number.dtype)
            out = self._paddle_diric.prob(value)
            [out] = self.executor.run(self.program, feed=feeds, fetch_list=[out])
            np.testing.assert_allclose(out, scipy.stats.dirichlet.pdf(random_number, self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))

    def test_log_prob(self):
        if False:
            for i in range(10):
                print('nop')
        with paddle.static.program_guard(self.program):
            random_number = np.random.rand(*self.concentration.shape)
            random_number = random_number / random_number.sum()
            feeds = dict(self.feeds, value=random_number)
            value = paddle.static.data('value', random_number.shape, random_number.dtype)
            out = self._paddle_diric.log_prob(value)
            [out] = self.executor.run(self.program, feed=feeds, fetch_list=[out])
            np.testing.assert_allclose(out, scipy.stats.dirichlet.logpdf(random_number, self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))

    def test_entropy(self):
        if False:
            i = 10
            return i + 15
        with paddle.static.program_guard(self.program):
            [out] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self._paddle_diric.entropy()])
            np.testing.assert_allclose(out, scipy.stats.dirichlet.entropy(self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))