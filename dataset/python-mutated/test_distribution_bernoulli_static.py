import unittest
import numpy as np
from distribution.config import ATOL, DEVICES, RTOL
from parameterize import TEST_CASE_NAME, parameterize_cls, parameterize_func, place
from test_distribution_bernoulli import BernoulliNumpy, _kstest, _sigmoid
import paddle
from paddle.distribution import Bernoulli
from paddle.distribution.kl import kl_divergence
np.random.seed(2023)
paddle.seed(2023)
paddle.enable_static()
default_dtype = paddle.get_default_dtype()

@place(DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'params'), [('params', (('probs_not_iterable', 0.3, 0.7, 1.0), ('probs_not_iterable_and_broadcast_for_value', 0.3, 0.7, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=default_dtype)), ('probs_tuple_0305', (0.3, 0.5), 0.7, 1.0), ('probs_tuple_03050104', ((0.3, 0.5), (0.1, 0.4)), 0.7, 1.0)))])
class BernoulliTestFeature(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        self.params_len = len(self.params)
        with paddle.static.program_guard(self.program):
            self.init_numpy_data(self.params)
            self.init_static_data(self.params)

    def init_numpy_data(self, params):
        if False:
            i = 10
            return i + 15
        self.mean_np = []
        self.variance_np = []
        self.log_prob_np = []
        self.prob_np = []
        self.cdf_np = []
        self.entropy_np = []
        self.kl_np = []
        for (_, probs, probs_other, value) in params:
            rv_np = BernoulliNumpy(probs)
            rv_np_other = BernoulliNumpy(probs_other)
            self.mean_np.append(rv_np.mean)
            self.variance_np.append(rv_np.variance)
            self.log_prob_np.append(rv_np.log_prob(value))
            self.prob_np.append(rv_np.prob(value))
            self.cdf_np.append(rv_np.cdf(value))
            self.entropy_np.append(rv_np.entropy())
            self.kl_np.append(rv_np.kl_divergence(rv_np_other))

    def init_static_data(self, params):
        if False:
            return 10
        with paddle.static.program_guard(self.program):
            rv_paddles = []
            rv_paddles_other = []
            values = []
            for (_, probs, probs_other, value) in params:
                if not isinstance(value, np.ndarray):
                    value = paddle.full([1], value, dtype=default_dtype)
                else:
                    value = paddle.to_tensor(value, place=self.place)
                rv_paddles.append(Bernoulli(probs=paddle.to_tensor(probs)))
                rv_paddles_other.append(Bernoulli(probs=paddle.to_tensor(probs_other)))
                values.append(value)
            results = self.executor.run(self.program, feed={}, fetch_list=[[rv_paddles[i].mean, rv_paddles[i].variance, rv_paddles[i].log_prob(values[i]), rv_paddles[i].prob(values[i]), rv_paddles[i].cdf(values[i]), rv_paddles[i].entropy(), rv_paddles[i].kl_divergence(rv_paddles_other[i]), kl_divergence(rv_paddles[i], rv_paddles_other[i])] for i in range(self.params_len)])
            self.mean_paddle = []
            self.variance_paddle = []
            self.log_prob_paddle = []
            self.prob_paddle = []
            self.cdf_paddle = []
            self.entropy_paddle = []
            self.kl_paddle = []
            self.kl_func_paddle = []
            for i in range(self.params_len):
                (_mean, _variance, _log_prob, _prob, _cdf, _entropy, _kl, _kl_func) = results[i * 8:(i + 1) * 8]
                self.mean_paddle.append(_mean)
                self.variance_paddle.append(_variance)
                self.log_prob_paddle.append(_log_prob)
                self.prob_paddle.append(_prob)
                self.cdf_paddle.append(_cdf)
                self.entropy_paddle.append(_entropy)
                self.kl_paddle.append(_kl)
                self.kl_func_paddle.append(_kl_func)

    def test_all(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(self.params_len):
            self._test_mean(i)
            self._test_variance(i)
            self._test_log_prob(i)
            self._test_prob(i)
            self._test_cdf(i)
            self._test_entropy(i)
            self._test_kl_divergence(i)

    def _test_mean(self, i):
        if False:
            while True:
                i = 10
        np.testing.assert_allclose(self.mean_np[i], self.mean_paddle[i], rtol=RTOL.get(default_dtype), atol=ATOL.get(default_dtype))

    def _test_variance(self, i):
        if False:
            print('Hello World!')
        np.testing.assert_allclose(self.variance_np[i], self.variance_paddle[i], rtol=RTOL.get(default_dtype), atol=ATOL.get(default_dtype))

    def _test_log_prob(self, i):
        if False:
            for i in range(10):
                print('nop')
        np.testing.assert_allclose(self.log_prob_np[i], self.log_prob_paddle[i], rtol=RTOL.get(default_dtype), atol=ATOL.get(default_dtype))

    def _test_prob(self, i):
        if False:
            i = 10
            return i + 15
        np.testing.assert_allclose(self.prob_np[i], self.prob_paddle[i], rtol=RTOL.get(default_dtype), atol=ATOL.get(default_dtype))

    def _test_cdf(self, i):
        if False:
            while True:
                i = 10
        np.testing.assert_allclose(self.cdf_np[i], self.cdf_paddle[i], rtol=RTOL.get(default_dtype), atol=ATOL.get(default_dtype))

    def _test_entropy(self, i):
        if False:
            print('Hello World!')
        np.testing.assert_allclose(self.entropy_np[i], self.entropy_paddle[i], rtol=RTOL.get(default_dtype), atol=ATOL.get(default_dtype))

    def _test_kl_divergence(self, i):
        if False:
            return 10
        np.testing.assert_allclose(self.kl_np[i], self.kl_paddle[i], rtol=RTOL.get(default_dtype), atol=ATOL.get(default_dtype))
        np.testing.assert_allclose(self.kl_np[i], self.kl_func_paddle[i], rtol=RTOL.get(default_dtype), atol=ATOL.get(default_dtype))

@place(DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'probs', 'shape', 'temperature', 'expected_shape'), [('probs_03', (0.3,), [100], 0.1, [100, 1]), ('probs_0305', (0.3, 0.5), [100], 0.1, [100, 2])])
class BernoulliTestSample(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            self.init_numpy_data(self.probs, self.shape)
            self.init_static_data(self.probs, self.shape, self.temperature)

    def init_numpy_data(self, probs, shape):
        if False:
            while True:
                i = 10
        self.rv_np = BernoulliNumpy(probs)
        self.sample_np = self.rv_np.sample(shape)

    def init_static_data(self, probs, shape, temperature):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(self.program):
            self.rv_paddle = Bernoulli(probs=paddle.to_tensor(probs))
            [self.sample_paddle, self.rsample_paddle] = self.executor.run(self.program, feed={}, fetch_list=[self.rv_paddle.sample(shape), self.rv_paddle.rsample(shape, temperature)])

    def test_sample(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            self.assertEqual(list(self.sample_paddle.shape), self.expected_shape)
            for i in range(len(self.probs)):
                self.assertTrue(_kstest(self.sample_np[..., i].reshape(-1), self.sample_paddle[..., i].reshape(-1)))

    def test_rsample(self):
        if False:
            while True:
                i = 10
        'Compare two samples from `rsample` method, one from scipy and another from paddle.'
        with paddle.static.program_guard(self.program):
            self.assertEqual(list(self.rsample_paddle.shape), self.expected_shape)
            for i in range(len(self.probs)):
                self.assertTrue(_kstest(self.sample_np[..., i].reshape(-1), (_sigmoid(self.rsample_paddle[..., i]) > 0.5).reshape(-1), self.temperature))

@place(DEVICES)
@parameterize_cls([TEST_CASE_NAME], ['BernoulliTestError'])
class BernoulliTestError(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

    @parameterize_func([(0,), ((0.3,),), ([0.3],), (np.array([0.3]),), (-1j + 1,), ('0',)])
    def test_bad_init_type(self, probs):
        if False:
            for i in range(10):
                print('nop')
        with paddle.static.program_guard(self.program):
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[Bernoulli(probs=probs)])

    @parameterize_func([(100,), (100.0,)])
    def test_bad_sample_shape_type(self, shape):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(0.3)
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.sample(shape)])
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.rsample(shape)])

    @parameterize_func([(1,)])
    def test_bad_rsample_temperature_type(self, temperature):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(0.3)
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.rsample([100], temperature)])

    @parameterize_func([(1,), (1.0,), ([1.0],), (1.0,), (np.array(1.0),)])
    def test_bad_value_type(self, value):
        if False:
            for i in range(10):
                print('nop')
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(0.3)
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.log_prob(value)])
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.prob(value)])
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.cdf(value)])

    @parameterize_func([(np.array(1.0),)])
    def test_bad_kl_other_type(self, other):
        if False:
            return 10
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(0.3)
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.kl_divergence(other)])

    @parameterize_func([(paddle.to_tensor([0.1, 0.2, 0.3]),)])
    def test_bad_broadcast(self, value):
        if False:
            i = 10
            return i + 15
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(paddle.to_tensor([0.3, 0.5]))
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.cdf(value)])
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.log_prob(value)])
            with self.assertRaises(TypeError):
                [_] = self.executor.run(self.program, feed={}, fetch_list=[rv.prob(value)])
if __name__ == '__main__':
    unittest.main()