"""Tests for ParameterizedTruncatedNormalOp."""
import functools
import math
import timeit
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

def _get_stddev_inside_bounds_before_using_randn(gpu):
    if False:
        for i in range(10):
            print('nop')
    if gpu:
        return 1.3
    else:
        return 1.7

class TruncatedNormalMoments:
    memoized_moments = None
    mean = None
    stddev = None
    minval = None
    maxval = None

    def __init__(self, mean, stddev, minval, maxval):
        if False:
            i = 10
            return i + 15
        self.memoized_moments = [1.0]
        self.mean = np.double(mean)
        self.stddev = np.double(stddev)
        self.minval = np.double(max(-10, minval))
        self.maxval = np.double(min(10, maxval))

    def __getitem__(self, moment):
        if False:
            for i in range(10):
                print('nop')
        'Calculates the truncated normal moments.\n\n    Args:\n      moment: The number for the moment.\n\n    Returns:\n      The value for the given moment.\n\n    Uses the recurrence relation described in:\n        http://www.smp.uq.edu.au/people/YoniNazarathy/teaching_projects\n            /studentWork/EricOrjebin_TruncatedNormalMoments.pdf\n    '
        assert moment > 0
        import scipy.stats
        dist = scipy.stats.norm(loc=self.mean, scale=self.stddev)
        for k in range(len(self.memoized_moments), moment + 1):
            m_k_minus_2 = self.memoized_moments[k - 2] if k > 1 else np.double(0.0)
            m_k_minus_1 = self.memoized_moments[k - 1]
            numerator = np.power(self.maxval, k - 1) * dist.pdf(self.maxval) - np.power(self.minval, k - 1) * dist.pdf(self.minval)
            denominator = dist.cdf(self.maxval) - dist.cdf(self.minval)
            m = (k - 1) * self.stddev ** 2 * m_k_minus_2 + self.mean * m_k_minus_1 - self.stddev * numerator / denominator
            assert abs(m) < 1e+50
            self.memoized_moments.append(m)
        return self.memoized_moments[moment]

def calculate_moments(samples, max_moment):
    if False:
        return 10
    moments = [0.0] * (max_moment + 1)
    for k in range(len(moments)):
        moments[k] = np.mean(samples ** k, axis=0)
    return moments

def z_test(real, expected, i, num_samples):
    if False:
        while True:
            i = 10
    numerical_error = 1e-06
    moment_mean = expected[i]
    moment_squared = expected[2 * i]
    moment_var = moment_squared - moment_mean * moment_mean
    error_per_moment = i * numerical_error
    total_variance = moment_var / float(num_samples) + error_per_moment
    return abs((real[i] - moment_mean) / math.sqrt(total_variance))

class ParameterizedTruncatedNormalTest(test.TestCase):
    z_limit = 6.0
    max_moment = 10

    def validateMoments(self, shape, mean, stddev, minval, maxval, use_stateless=False, seed=1618):
        if False:
            for i in range(10):
                print('nop')
        try:
            random_seed.set_random_seed(seed)
            with self.cached_session():
                if use_stateless:
                    new_seed = random_ops.random_uniform([2], seed=seed, minval=0, maxval=2 ** 31 - 1, dtype=np.int32)
                    samples = stateless.stateless_parameterized_truncated_normal(shape, new_seed, mean, stddev, minval, maxval).eval()
                else:
                    samples = random_ops.parameterized_truncated_normal(shape, mean, stddev, minval, maxval).eval()
                assert (~np.isnan(samples)).all()
            moments = calculate_moments(samples, self.max_moment)
            expected_moments = TruncatedNormalMoments(mean, stddev, minval, maxval)
            num_samples = functools.reduce(lambda x, y: x * y, shape, 1)
            for i in range(1, len(moments)):
                self.assertLess(z_test(moments, expected_moments, i, num_samples), self.z_limit)
        except ImportError as e:
            tf_logging.warn('Cannot test truncated normal op: %s' % str(e))

    def validateKolmogorovSmirnov(self, shape, mean, stddev, minval, maxval, use_stateless=False, seed=1618):
        if False:
            print('Hello World!')
        try:
            import scipy.stats
            random_seed.set_random_seed(seed)
            with self.cached_session():
                if use_stateless:
                    new_seed = random_ops.random_uniform([2], seed=seed, minval=0, maxval=2 ** 31 - 1, dtype=np.int32)
                    samples = stateless.stateless_parameterized_truncated_normal(shape, new_seed, mean, stddev, minval, maxval).eval()
                else:
                    samples = random_ops.parameterized_truncated_normal(shape, mean, stddev, minval, maxval).eval()
            assert (~np.isnan(samples)).all()
            minval = max(mean - stddev * 10, minval)
            maxval = min(mean + stddev * 10, maxval)
            dist = scipy.stats.norm(loc=mean, scale=stddev)
            cdf_min = dist.cdf(minval)
            cdf_max = dist.cdf(maxval)

            def truncated_cdf(x):
                if False:
                    print('Hello World!')
                return np.clip((dist.cdf(x) - cdf_min) / (cdf_max - cdf_min), 0.0, 1.0)
            pvalue = scipy.stats.kstest(samples, truncated_cdf)[1]
            self.assertGreater(pvalue, 1e-10)
        except ImportError as e:
            tf_logging.warn('Cannot test truncated normal op: %s' % str(e))

    @test_util.run_deprecated_v1
    def testDefaults(self):
        if False:
            while True:
                i = 10
        self.validateMoments([int(100000.0)], 0.0, 1.0, -2.0, 2.0)
        self.validateMoments([int(100000.0)], 0.0, 1.0, -2.0, 2.0, use_stateless=True)

    @test_util.run_deprecated_v1
    def testShifted(self):
        if False:
            while True:
                i = 10
        self.validateMoments([int(100000.0)], -1.0, 1.0, -2.0, 2.0)
        self.validateMoments([int(100000.0)], -1.0, 1.0, -2.0, 2.0, use_stateless=True)

    @test_util.run_deprecated_v1
    def testRightTail(self):
        if False:
            print('Hello World!')
        self.validateMoments([int(100000.0)], 0.0, 1.0, 4.0, np.infty)
        self.validateMoments([int(100000.0)], 0.0, 1.0, 4.0, np.infty, use_stateless=True)

    @test_util.run_deprecated_v1
    def testLeftTail(self):
        if False:
            while True:
                i = 10
        self.validateMoments([int(100000.0)], 0.0, 1.0, -np.infty, -4.0)
        self.validateMoments([int(100000.0)], 0.0, 1.0, -np.infty, -4.0, use_stateless=True)

    @test_util.run_deprecated_v1
    def testLeftTailTwoSidedBounds(self):
        if False:
            print('Hello World!')
        self.validateMoments([int(100000.0)], 0.0, 1.0, -6.0, -3.0)
        self.validateMoments([int(100000.0)], 0.0, 1.0, -6.0, -3.0, use_stateless=True)

    @test_util.run_deprecated_v1
    @test_util.disable_xla('Low probability region')
    def testTwoSidedLeftTailShifted(self):
        if False:
            for i in range(10):
                print('nop')
        self.validateKolmogorovSmirnov([int(100000.0)], 6.0, 1.0, -1.0, 1.0)
        self.validateKolmogorovSmirnov([int(100000.0)], 6.0, 1.0, -1.0, 1.0, use_stateless=True)

    @test_util.run_deprecated_v1
    @test_util.disable_xla('Low probability region')
    def testRightTailShifted(self):
        if False:
            for i in range(10):
                print('nop')
        self.validateMoments([int(100000.0)], -5.0, 1.0, 2.0, np.infty)
        self.validateMoments([int(100000.0)], -5.0, 1.0, 2.0, np.infty, use_stateless=True)

    @test_util.run_deprecated_v1
    def testTruncateOnLeft_entireTailOnRight(self):
        if False:
            i = 10
            return i + 15
        self.validateKolmogorovSmirnov([int(100000.0)], 10.0, 1.0, 4.0, np.infty)
        self.validateKolmogorovSmirnov([int(100000.0)], 10.0, 1.0, 4.0, np.infty, use_stateless=True)

    @test_util.run_deprecated_v1
    def testTruncateOnRight_entireTailOnLeft(self):
        if False:
            print('Hello World!')
        self.validateKolmogorovSmirnov([int(100000.0)], -8, 1.0, -np.infty, -4.0)
        self.validateKolmogorovSmirnov([int(100000.0)], -8.0, 1.0, -np.infty, -4.0, use_stateless=True)

    @test_util.run_deprecated_v1
    def testSmallStddev(self):
        if False:
            for i in range(10):
                print('nop')
        self.validateKolmogorovSmirnov([int(100000.0)], 0.0, 0.1, 0.05, 0.1)
        self.validateKolmogorovSmirnov([int(100000.0)], 0.0, 0.1, 0.05, 0.1, use_stateless=True)

    @test_util.run_deprecated_v1
    def testSamplingWithSmallStdDevFarFromBound(self):
        if False:
            for i in range(10):
                print('nop')
        sample_op = random_ops.parameterized_truncated_normal(shape=(int(100000.0),), means=0.8, stddevs=0.05, minvals=-1.0, maxvals=1.0)
        new_seed = random_ops.random_uniform([2], seed=1234, minval=0, maxval=2 ** 31 - 1, dtype=np.int32)
        sample_op_stateless = stateless.stateless_parameterized_truncated_normal(shape=(int(100000.0),), seed=new_seed, means=0.8, stddevs=0.05, minvals=-1.0, maxvals=1.0)
        with self.session() as sess:
            (samples, samples_stateless) = sess.run([sample_op, sample_op_stateless])
            assert (~np.isnan(samples)).all()
            assert (~np.isnan(samples_stateless)).all()
            self.assertAllGreater(samples, 0.0)
            self.assertAllGreater(samples_stateless, 0.0)

    def testShapeTypes(self):
        if False:
            for i in range(10):
                print('nop')
        for shape_dtype in [np.int32, np.int64]:
            shape = np.array([1000], dtype=shape_dtype)
            sample_op = random_ops.parameterized_truncated_normal(shape=shape, means=0.0, stddevs=0.1, minvals=-1.0, maxvals=1.0)
            new_seed = random_ops.random_uniform([2], seed=1234, minval=0, maxval=2 ** 31 - 1, dtype=np.int32)
            sample_op_stateless = stateless.stateless_parameterized_truncated_normal(shape=shape, seed=new_seed, means=0.0, stddevs=0.1, minvals=-1.0, maxvals=1.0)
            samples = self.evaluate(sample_op)
            stateless_samples = self.evaluate(sample_op_stateless)
            self.assertAllEqual(samples.shape, shape)
            self.assertAllEqual(stateless_samples.shape, shape)

    def testStatelessParameterizedTruncatedNormalHasGrads(self):
        if False:
            i = 10
            return i + 15
        mean = variables.Variable(0.01)
        stddev = variables.Variable(1.0)
        minval = variables.Variable(-1.0)
        maxval = variables.Variable(1.0)
        with self.cached_session() as sess:
            with backprop.GradientTape(persistent=True) as tape:
                samples = stateless.stateless_parameterized_truncated_normal([1], [1, 2], mean, stddev, minval, maxval)
            sess.run(variables.variables_initializer([mean, stddev, minval, maxval]))
            ([mean_grad, std_grad], mean_actual_grad, std_actual_grad) = sess.run([tape.gradient(samples, [mean, stddev]), array_ops.ones_like(mean), (samples - mean) / stddev])
            self.assertAllClose(mean_grad, mean_actual_grad)
            self.assertAllClose(std_grad, std_actual_grad[0])
            try:
                import scipy.stats
                truncnorm = scipy.stats.truncnorm(a=-1.0, b=1.0, loc=0.0, scale=1.0)
                (samples_np, [minval_grad, maxval_grad]) = sess.run([samples, tape.gradient(samples, [minval, maxval])])
                sample_cdf = truncnorm.cdf(samples_np)
                scipy_maxval_grad = np.exp(0.5 * (samples_np ** 2 - ((1.0 - 0.01) / 1.0) ** 2) + np.log(sample_cdf))
                scipy_minval_grad = np.exp(0.5 * (samples_np ** 2 - ((-1.0 - 0.01) / 1.0) ** 2) + np.log1p(-sample_cdf))
                self.assertAllClose(minval_grad, scipy_minval_grad[0], rtol=0.01)
                self.assertAllClose(maxval_grad, scipy_maxval_grad[0], rtol=0.01)
            except ImportError as e:
                tf_logging.warn('Cannot test truncated normal op: %s' % str(e))

    @test_util.run_deprecated_v1
    def testSamplingAtRandnSwitchover(self):
        if False:
            return 10
        use_gpu = test.is_gpu_available()
        stddev_inside_bounds_before_using_randn = _get_stddev_inside_bounds_before_using_randn(use_gpu)
        epsilon = 0.001
        self.validateMoments(shape=[int(1000000.0)], mean=0.0, stddev=1.0, minval=-epsilon, maxval=stddev_inside_bounds_before_using_randn - epsilon)
        self.validateMoments(shape=[int(1000000.0)], mean=0.0, stddev=1.0, minval=-epsilon, maxval=stddev_inside_bounds_before_using_randn + epsilon)
        self.validateMoments(shape=[int(1000000.0)], mean=0.0, stddev=1.0, minval=-epsilon, maxval=stddev_inside_bounds_before_using_randn - epsilon, use_stateless=True)
        self.validateMoments(shape=[int(1000000.0)], mean=0.0, stddev=1.0, minval=-epsilon, maxval=stddev_inside_bounds_before_using_randn + epsilon, use_stateless=True)

def parameterized_vs_naive(shape, num_iters, use_gpu=False):
    if False:
        return 10
    np.random.seed(1618)
    optimizer_options = config_pb2.OptimizerOptions(opt_level=config_pb2.OptimizerOptions.L0)
    config = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(optimizer_options=optimizer_options))
    with session.Session(config=config) as sess:
        with ops.device('/cpu:0' if not use_gpu else None):
            param_op = control_flow_ops.group(random_ops.parameterized_truncated_normal(shape))
            naive_op = control_flow_ops.group(random_ops.truncated_normal(shape))
        sess.run(param_op)
        sess.run(param_op)
        param_dt = timeit.timeit(lambda : sess.run(param_op), number=num_iters)
        sess.run(naive_op)
        sess.run(naive_op)
        naive_dt = timeit.timeit(lambda : sess.run(naive_op), number=num_iters)
        return (param_dt, naive_dt)

def randn_sampler_switchover(shape, num_iters, use_gpu=False):
    if False:
        for i in range(10):
            print('nop')
    stddev_inside_bounds_before_using_randn = _get_stddev_inside_bounds_before_using_randn(use_gpu)
    epsilon = 0.001
    np.random.seed(1618)
    optimizer_options = config_pb2.OptimizerOptions(opt_level=config_pb2.OptimizerOptions.L0)
    config = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(optimizer_options=optimizer_options))
    with session.Session(config=config) as sess:
        with ops.device('/cpu:0' if not use_gpu else '/gpu:0'):
            uniform_sampler_op = control_flow_ops.group(random_ops.parameterized_truncated_normal(shape, means=0.0, stddevs=1.0, minvals=-stddev_inside_bounds_before_using_randn + epsilon, maxvals=0.01))
            randn_sampler_op = control_flow_ops.group(random_ops.parameterized_truncated_normal(shape, means=0.0, stddevs=1.0, minvals=-stddev_inside_bounds_before_using_randn - epsilon, maxvals=0.01))
        sess.run(uniform_sampler_op)
        sess.run(uniform_sampler_op)
        uniform_dt = timeit.timeit(lambda : sess.run(uniform_sampler_op), number=num_iters)
        sess.run(randn_sampler_op)
        sess.run(randn_sampler_op)
        randn_dt = timeit.timeit(lambda : sess.run(randn_sampler_op), number=num_iters)
        return (randn_dt, uniform_dt)

class TruncatedNormalBenchmark(test.Benchmark):

    def benchmarkParameterizedOpVsNaiveOpCpu(self):
        if False:
            i = 10
            return i + 15
        self._benchmarkParameterizedOpVsNaiveOp(False)

    def benchmarkParameterizedOpVsNaiveOpGpu(self):
        if False:
            for i in range(10):
                print('nop')
        self._benchmarkParameterizedOpVsNaiveOp(True)

    def _benchmarkParameterizedOpVsNaiveOp(self, use_gpu):
        if False:
            print('Hello World!')
        num_iters = 50
        print('Composition of new ParameterizedTruncatedNormalOp vs. naive TruncatedNormalOp [%d iters]' % num_iters)
        print('Shape\tsec(parameterized)\tsec(naive)\tspeedup')
        for shape in [[10000, 100], [1000, 1000], [1000000], [100, 100, 100], [20, 20, 20, 20]]:
            (p_dt, n_dt) = parameterized_vs_naive(shape, num_iters, use_gpu)
            print('%s\t%.3f\t%.3f\t%.2f' % (shape, p_dt, n_dt, p_dt / n_dt))
            shape_str = '-'.join(map(str, shape))
            self.report_benchmark(name='parameterized_shape' + shape_str, iters=num_iters, wall_time=p_dt)
            self.report_benchmark(name='naive_shape' + shape_str, iters=num_iters, wall_time=n_dt)

    def benchmarkRandnSamplerCPU(self):
        if False:
            print('Hello World!')
        self._benchmarkRandnSampler(False)

    def benchmarkRandnSamplerGPU(self):
        if False:
            i = 10
            return i + 15
        self._benchmarkRandnSampler(True)

    def _benchmarkRandnSampler(self, use_gpu):
        if False:
            print('Hello World!')
        num_iters = 100
        shape = [int(1000000.0)]
        (randn_dt, uniform_dt) = randn_sampler_switchover(shape, num_iters, use_gpu)
        print('Randn Sampler vs uniform samplers [%d iters]\t%.4f\t%.4f' % (num_iters, randn_dt, uniform_dt))
        gpu_str = '_gpu' if use_gpu else '_cpu'
        self.report_benchmark(name='randn_sampler' + gpu_str, iters=num_iters, wall_time=randn_dt)
        self.report_benchmark(name='uniform_sampler' + gpu_str, iters=num_iters, wall_time=uniform_dt)
if __name__ == '__main__':
    test.main()