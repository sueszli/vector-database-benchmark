import unittest
import mock_data as mock
import numpy as np
import parameterize
from distribution import config
import paddle
np.random.seed(2022)

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'dist'), [('test-mock-exp', mock.Exponential(rate=paddle.rand([100, 200, 99], dtype=config.DEFAULT_DTYPE)))])
class TestExponentialFamily(unittest.TestCase):

    def test_entropy(self):
        if False:
            i = 10
            return i + 15
        np.testing.assert_allclose(self.dist.entropy(), paddle.distribution.ExponentialFamily.entropy(self.dist), rtol=config.RTOL.get(config.DEFAULT_DTYPE), atol=config.ATOL.get(config.DEFAULT_DTYPE))

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((config.TEST_CASE_NAME, 'dist'), [('test-dummy', mock.DummyExpFamily(0.5, 0.5)), ('test-dirichlet', paddle.distribution.Dirichlet(paddle.to_tensor(parameterize.xrand()))), ('test-beta', paddle.distribution.Beta(paddle.to_tensor(parameterize.xrand()), paddle.to_tensor(parameterize.xrand())))])
class TestExponentialFamilyException(unittest.TestCase):

    def test_entropy_exception(self):
        if False:
            print('Hello World!')
        with self.assertRaises(NotImplementedError):
            paddle.distribution.ExponentialFamily.entropy(self.dist)
if __name__ == '__main__':
    unittest.main()