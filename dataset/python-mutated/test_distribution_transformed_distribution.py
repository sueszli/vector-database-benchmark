import unittest
import numpy as np
import parameterize as param
from distribution import config
import paddle

@param.place(config.DEVICES)
@param.param_cls((param.TEST_CASE_NAME, 'base', 'transforms'), [('base_normal', paddle.distribution.Normal(0.0, 1.0), [paddle.distribution.ExpTransform()])])
class TestIndependent(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._t = paddle.distribution.TransformedDistribution(self.base, self.transforms)

    def _np_sum_rightmost(self, value, n):
        if False:
            return 10
        return np.sum(value, tuple(range(-n, 0))) if n > 0 else value

    def test_log_prob(self):
        if False:
            for i in range(10):
                print('nop')
        value = paddle.to_tensor([0.5])
        np.testing.assert_allclose(self.simple_log_prob(value, self.base, self.transforms), self._t.log_prob(value), rtol=config.RTOL.get(str(value.numpy().dtype)), atol=config.ATOL.get(str(value.numpy().dtype)))

    def simple_log_prob(self, value, base, transforms):
        if False:
            i = 10
            return i + 15
        log_prob = 0.0
        y = value
        for t in reversed(transforms):
            x = t.inverse(y)
            log_prob = log_prob - t.forward_log_det_jacobian(x)
            y = x
        log_prob += base.log_prob(y)
        return log_prob

    def test_sample(self):
        if False:
            print('Hello World!')
        shape = [5, 10, 8]
        expected_shape = (5, 10, 8)
        data = self._t.sample(shape)
        self.assertEqual(tuple(data.shape), expected_shape)
        self.assertEqual(data.dtype, self.base.loc.dtype)

    def test_rsample(self):
        if False:
            print('Hello World!')
        shape = [5, 10, 8]
        expected_shape = (5, 10, 8)
        data = self._t.rsample(shape)
        self.assertEqual(tuple(data.shape), expected_shape)
        self.assertEqual(data.dtype, self.base.loc.dtype)
if __name__ == '__main__':
    unittest.main()