import unittest
import mock_data as mock
import numpy as np
import parameterize
from distribution import config
import paddle
np.random.seed(2022)
paddle.enable_static()

@parameterize.place(config.DEVICES)
class TestExponentialFamily(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor()
        with paddle.static.program_guard(self.program):
            rate_np = parameterize.xrand((100, 200, 99))
            rate = paddle.static.data('rate', rate_np.shape, rate_np.dtype)
            self.mock_dist = mock.Exponential(rate)
            self.feeds = {'rate': rate_np}

    def test_entropy(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            [out1, out2] = self.executor.run(self.program, feed=self.feeds, fetch_list=[self.mock_dist.entropy(), paddle.distribution.ExponentialFamily.entropy(self.mock_dist)])
            np.testing.assert_allclose(out1, out2, rtol=config.RTOL.get(config.DEFAULT_DTYPE), atol=config.ATOL.get(config.DEFAULT_DTYPE))

    def test_entropy_exception(self):
        if False:
            i = 10
            return i + 15
        with paddle.static.program_guard(self.program):
            with self.assertRaises(NotImplementedError):
                paddle.distribution.ExponentialFamily.entropy(mock.DummyExpFamily(0.5, 0.5))