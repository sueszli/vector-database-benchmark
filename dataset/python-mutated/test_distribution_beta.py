import numbers
import unittest
import numpy as np
import scipy.stats
from distribution.config import ATOL, DEVICES, RTOL
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand
import paddle
np.random.seed(2022)

@place(DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'alpha', 'beta'), [('test-scale', 1.0, 2.0), ('test-tensor', xrand(), xrand()), ('test-broadcast', xrand((2, 1)), xrand((2, 5)))])
class TestBeta(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        (alpha, beta) = (self.alpha, self.beta)
        if not isinstance(self.alpha, numbers.Real):
            alpha = paddle.to_tensor(self.alpha)
        if not isinstance(self.beta, numbers.Real):
            beta = paddle.to_tensor(self.beta)
        self._paddle_beta = paddle.distribution.Beta(alpha, beta)

    def test_mean(self):
        if False:
            for i in range(10):
                print('nop')
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(self._paddle_beta.mean, scipy.stats.beta.mean(self.alpha, self.beta), rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)), atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_variance(self):
        if False:
            while True:
                i = 10
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(self._paddle_beta.variance, scipy.stats.beta.var(self.alpha, self.beta), rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)), atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_prob(self):
        if False:
            i = 10
            return i + 15
        value = [np.random.rand(*self._paddle_beta.alpha.shape)]
        for v in value:
            with paddle.base.dygraph.guard(self.place):
                np.testing.assert_allclose(self._paddle_beta.prob(paddle.to_tensor(v)), scipy.stats.beta.pdf(v, self.alpha, self.beta), rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)), atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_log_prob(self):
        if False:
            while True:
                i = 10
        value = [np.random.rand(*self._paddle_beta.alpha.shape)]
        for v in value:
            with paddle.base.dygraph.guard(self.place):
                np.testing.assert_allclose(self._paddle_beta.log_prob(paddle.to_tensor(v)), scipy.stats.beta.logpdf(v, self.alpha, self.beta), rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)), atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_entropy(self):
        if False:
            print('Hello World!')
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(self._paddle_beta.entropy(), scipy.stats.beta.entropy(self.alpha, self.beta), rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)), atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_sample_shape(self):
        if False:
            while True:
                i = 10
        cases = [{'input': [], 'expect': [] + paddle.squeeze(self._paddle_beta.alpha).shape}, {'input': [2, 3], 'expect': [2, 3] + paddle.squeeze(self._paddle_beta.alpha).shape}]
        for case in cases:
            self.assertTrue(self._paddle_beta.sample(case.get('input')).shape == case.get('expect'))

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            array = np.array([], dtype=np.float32)
            x = paddle.to_tensor(np.reshape(array, [0]), dtype='int32')
            paddle.distribution.Beta(alpha=x, beta=x)
if __name__ == '__main__':
    unittest.main()