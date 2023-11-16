import unittest
import numpy as np
import parameterize as param
import scipy.stats
from distribution.config import ATOL, DEVICES, RTOL
import paddle
np.random.seed(2022)

@param.place(DEVICES)
@param.param_cls((param.TEST_CASE_NAME, 'concentration'), [('test-one-dim', param.xrand((89,)))])
class TestDirichlet(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._paddle_diric = paddle.distribution.Dirichlet(paddle.to_tensor(self.concentration))

    def test_mean(self):
        if False:
            print('Hello World!')
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(self._paddle_diric.mean, scipy.stats.dirichlet.mean(self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))

    def test_variance(self):
        if False:
            return 10
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(self._paddle_diric.variance, scipy.stats.dirichlet.var(self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))

    def test_prob(self):
        if False:
            for i in range(10):
                print('nop')
        value = [np.random.rand(*self.concentration.shape)]
        value = [v / v.sum() for v in value]
        for v in value:
            with paddle.base.dygraph.guard(self.place):
                np.testing.assert_allclose(self._paddle_diric.prob(paddle.to_tensor(v)), scipy.stats.dirichlet.pdf(v, self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))

    def test_log_prob(self):
        if False:
            i = 10
            return i + 15
        value = [np.random.rand(*self.concentration.shape)]
        value = [v / v.sum() for v in value]
        for v in value:
            with paddle.base.dygraph.guard(self.place):
                np.testing.assert_allclose(self._paddle_diric.log_prob(paddle.to_tensor(v)), scipy.stats.dirichlet.logpdf(v, self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))

    def test_entropy(self):
        if False:
            i = 10
            return i + 15
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(self._paddle_diric.entropy(), scipy.stats.dirichlet.entropy(self.concentration), rtol=RTOL.get(str(self.concentration.dtype)), atol=ATOL.get(str(self.concentration.dtype)))

    def test_natural_parameters(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(isinstance(self._paddle_diric._natural_parameters, tuple))

    def test_log_normalizer(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(np.all(self._paddle_diric._log_normalizer(paddle.to_tensor(param.xrand((100, 100, 100)))).numpy() < 0.0))

    @param.place(DEVICES)
    @param.param_cls((param.TEST_CASE_NAME, 'concentration'), [('test-zero-dim', np.array(1.0))])
    class TestDirichletException(unittest.TestCase):

        def TestInit(self):
            if False:
                return 10
            with self.assertRaises(ValueError):
                paddle.distribution.Dirichlet(paddle.squeeze(self.concentration))
if __name__ == '__main__':
    unittest.main()