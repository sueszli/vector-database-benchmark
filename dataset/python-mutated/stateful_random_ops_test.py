"""Tests for stateful random-number generation ops."""
import itertools
import os
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import device_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util as random_test_util
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import stateful_random_ops as random
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
FLAGS = flags.FLAGS

def xla_device():
    if False:
        for i in range(10):
            print('nop')
    devices = device_lib.list_local_devices()

    def find_type(device_type):
        if False:
            return 10
        for d in devices:
            if d.device_type == device_type:
                return d
        return None
    d = find_type('TPU') or find_type('XLA_GPU') or find_type('XLA_CPU')
    if d is None:
        raise ValueError("Can't find any XLA device. Available devices:\n%s" % devices)
    return d

def xla_device_name():
    if False:
        while True:
            i = 10
    return str(xla_device().name)
ALGS = [random_ops_util.Algorithm.PHILOX.value, random_ops_util.Algorithm.THREEFRY.value, random_ops_util.Algorithm.AUTO_SELECT.value]
INTS = [dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64]
FLOATS = [dtypes.bfloat16, dtypes.float32, dtypes.float64]

class StatefulRandomOpsTest(xla_test.XLATestCase, parameterized.TestCase):
    """Test cases for stateful random-number generator operators."""

    @parameterized.parameters(ALGS)
    def testSimple(self, alg):
        if False:
            for i in range(10):
                print('nop')
        'A simple test.'
        with ops.device(xla_device_name()):
            gen = random.Generator.from_seed(seed=0, alg=alg)
            gen.normal(shape=(3,))
            gen.uniform(shape=(3,), minval=0, maxval=10, dtype=dtypes.uint32)
            gen.uniform_full_int(shape=(3,))

    @parameterized.parameters(ALGS)
    def testDefun(self, alg):
        if False:
            i = 10
            return i + 15
        'Test for defun.'
        with ops.device(xla_device_name()):
            gen = random.Generator.from_seed(seed=0, alg=alg)

            @def_function.function
            def f():
                if False:
                    return 10
                x = gen.normal(shape=(3,))
                y = gen.uniform(shape=(3,), minval=0, maxval=10, dtype=dtypes.uint32)
                z = gen.uniform_full_int(shape=(3,))
                return (x, y, z)
            f()

    def _compareToKnownOutputs(self, g, counter, key, expect):
        if False:
            while True:
                i = 10
        'Compares against known outputs for specific counter and key inputs.'

        def uint32s_to_uint64(a, b):
            if False:
                print('Hello World!')
            return b << 32 | a

        def uint32s_to_uint64s(ls):
            if False:
                i = 10
                return i + 15
            return [uint32s_to_uint64(ls[2 * i], ls[2 * i + 1]) for i in range(len(ls) // 2)]
        ctr_len = len(counter)
        counter = uint32s_to_uint64s(counter)
        key = uint32s_to_uint64s(key)
        state = counter + key
        g.reset(state)
        got = g.uniform_full_int(shape=(ctr_len,), dtype=dtypes.uint32)
        self.assertAllEqual(expect, got)
        g.reset(state)
        got = g.uniform_full_int(shape=(ctr_len // 2,), dtype=dtypes.uint64)
        self.assertAllEqual(uint32s_to_uint64s(expect), got)

    def testThreefry2x32(self):
        if False:
            print('Hello World!')
        'Tests ThreeFry2x32 conforms to known results.\n    '
        with ops.device(xla_device_name()):
            g = random.Generator.from_seed(seed=0, alg=random.RNG_ALG_THREEFRY)
            self._compareToKnownOutputs(g, [0, 0], [0, 0], [1797259609, 2579123966])
            self._compareToKnownOutputs(g, [4294967295, 4294967295], [4294967295, 4294967295], [481924860, 3137350631])
            self._compareToKnownOutputs(g, [608135816, 2242054355], [320440878, 57701188], [3297917596, 1212020640])

    def testPhilox4x32(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests Philox4x32 conforms to known results.\n    '
        with ops.device(xla_device_name()):
            g = random.Generator.from_seed(seed=0, alg=random.RNG_ALG_PHILOX)
            self._compareToKnownOutputs(g, [0, 0, 0, 0], [0, 0], [1713891541, 3781805453, 3159862348, 2600524760])
            self._compareToKnownOutputs(g, [4294967295, 4294967295, 4294967295, 4294967295], [4294967295, 4294967295], [1083123565, 1103641358, 2718681030, 1834242557])
            self._compareToKnownOutputs(g, [608135816, 2242054355, 320440878, 57701188], [2752067618, 698298832], [3513581065, 2499661035, 1342301216, 605187745])

    @parameterized.parameters(INTS)
    def testXLAEqualsCPU(self, dtype):
        if False:
            print('Hello World!')
        'Tests that XLA and CPU kernels generate the same integers.'
        seed = 1234
        shape = [315, 49]
        with ops.device('/device:CPU:0'):
            cpu_gen = random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
        with ops.device(xla_device_name()):
            xla_gen = random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
        for _ in range(5):
            with ops.device('/device:CPU:0'):
                cpu = cpu_gen.uniform_full_int(shape=shape, dtype=dtype)
                cpu_gen.skip(100)
            with ops.device(xla_device_name()):
                xla = xla_gen.uniform_full_int(shape=shape, dtype=dtype)
                xla_gen.skip(100)
            self.assertAllEqual(cpu, xla)
            self.assertAllEqual(cpu_gen.state, xla_gen.state)

    def testXLAEqualsCPUAroundCounterOverflow(self):
        if False:
            return 10
        'Tests XLA and CPU kernels generate the same integers in overflow case.\n\n       Specifically this tests the case where the counter is incremented past\n       what can fit within 64 bits of the 128 bit Philox counter.\n    '
        dtype = dtypes.uint64
        seed = 2 ** 64 - 10
        shape = [315, 49]
        with ops.device('/device:CPU:0'):
            cpu_gen = random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
        with ops.device(xla_device_name()):
            xla_gen = random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
        for _ in range(5):
            with ops.device('/device:CPU:0'):
                cpu = cpu_gen.uniform_full_int(shape=shape, dtype=dtype)
                cpu_gen.skip(100)
            with ops.device(xla_device_name()):
                xla = xla_gen.uniform_full_int(shape=shape, dtype=dtype)
                xla_gen.skip(100)
            self.assertAllEqual(cpu, xla)
            self.assertAllEqual(cpu_gen.state, xla_gen.state)
        self.assertAllEqual(cpu, xla)

    def _testRngIsNotConstant(self, rng, dtype):
        if False:
            i = 10
            return i + 15
        x = rng(dtype).numpy()
        y = rng(dtype).numpy()
        self.assertFalse(np.array_equal(x, y))

    def check_dtype(self, dtype):
        if False:
            print('Hello World!')
        device = xla_device()
        if device.device_type == 'TPU' and dtype == dtypes.float64:
            self.skipTest("TPU doesn't support float64.")

    @parameterized.parameters(list(itertools.product(ALGS, INTS + FLOATS)))
    def testUniformIsNotConstant(self, alg, dtype):
        if False:
            while True:
                i = 10
        self.check_dtype(dtype)
        with ops.device(xla_device_name()):
            gen = random.Generator.from_seed(seed=1234, alg=alg)

            def rng(dtype):
                if False:
                    print('Hello World!')
                maxval = dtype.max
                return gen.uniform(shape=[2], dtype=dtype, maxval=maxval)
            self._testRngIsNotConstant(rng, dtype)

    @parameterized.parameters(list(itertools.product(ALGS, FLOATS)))
    def testNormalIsNotConstant(self, alg, dtype):
        if False:
            return 10
        self.check_dtype(dtype)
        with ops.device(xla_device_name()):
            gen = random.Generator.from_seed(seed=1234, alg=alg)

            def rng(dtype):
                if False:
                    while True:
                        i = 10
                return gen.normal(shape=[2], dtype=dtype)
            self._testRngIsNotConstant(rng, dtype)

    @parameterized.parameters(list(itertools.product(ALGS, INTS + FLOATS)))
    def testUniformIsInRange(self, alg, dtype):
        if False:
            return 10
        self.check_dtype(dtype)
        minval = 2
        maxval = 33
        size = 1000
        with ops.device(xla_device_name()):
            gen = random.Generator.from_seed(seed=1234, alg=alg)
            x = gen.uniform(shape=[size], dtype=dtype, minval=minval, maxval=maxval).numpy()
            self.assertTrue(np.all(x >= minval))
            self.assertTrue(np.all(x <= maxval))

    @parameterized.parameters(list(itertools.product(ALGS, FLOATS)))
    def testNormalIsFinite(self, alg, dtype):
        if False:
            while True:
                i = 10
        self.check_dtype(dtype)
        with ops.device(xla_device_name()):
            gen = random.Generator.from_seed(seed=1234, alg=alg)
            x = gen.normal(shape=[10000], dtype=dtype).numpy()
            self.assertTrue(np.all(np.isfinite(x)))

    @parameterized.parameters(list(itertools.product(ALGS, INTS + FLOATS, (12, 13, 123, 4321))))
    def testDistributionOfUniform(self, alg, dtype, seed):
        if False:
            for i in range(10):
                print('nop')
        "Use Pearson's Chi-squared test to test for uniformity."
        self.check_dtype(dtype)
        three_fry = random_ops_util.Algorithm.THREEFRY.value
        auto_select = random_ops_util.Algorithm.AUTO_SELECT.value
        is_tpu = xla_device().device_type == 'TPU'
        is_megacore = 'megacore' in os.environ.get('TEST_TARGET', '').lower()
        if (alg, dtype, seed) in [(three_fry, dtypes.int64, 123), (three_fry, dtypes.uint64, 123)] or (is_tpu and (alg, dtype, seed) in [(auto_select, dtypes.int64, 123), (auto_select, dtypes.uint64, 123)]) or (is_megacore and (alg, dtype, seed) in [(auto_select, dtypes.int32, 123), (auto_select, dtypes.uint32, 123), (auto_select, dtypes.int32, 12), (auto_select, dtypes.uint32, 12)]):
            self.skipTest('This (device, alg, dtype, seed) combination fails (b/244649364).')
        with ops.device(xla_device_name()):
            n = 1000
            gen = random.Generator.from_seed(seed=seed, alg=alg)
            maxval = 1
            if dtype.is_integer:
                maxval = 100
            t = gen.uniform(shape=[n], maxval=maxval, dtype=dtype)
            x = t.numpy().astype(float)
            if maxval > 1:
                x = x / maxval
            val = random_test_util.chi_squared(x, 10)
            self.assertLess(val, 16.92)

    @parameterized.parameters(list(itertools.product(ALGS, FLOATS)))
    def testDistributionOfNormal(self, alg, dtype):
        if False:
            for i in range(10):
                print('nop')
        'Use Anderson-Darling test to test distribution appears normal.'
        self.check_dtype(dtype)
        with ops.device(xla_device_name()):
            n = 1000
            gen = random.Generator.from_seed(seed=1234, alg=alg)
            x = gen.normal(shape=[n], dtype=dtype).numpy()
            self.assertLess(random_test_util.anderson_darling(x.astype(float)), 2.492)

    @parameterized.parameters(list(itertools.product(ALGS, FLOATS)))
    def testTruncatedNormal(self, alg, dtype):
        if False:
            return 10
        self.check_dtype(dtype)
        with ops.device(xla_device_name()):
            gen = random.Generator.from_seed(seed=123, alg=alg)
            n = 100000
            y = gen.truncated_normal(shape=[n], dtype=dtype).numpy()
            random_test_util.test_truncated_normal(self.assertEqual, self.assertAllClose, n, y, mean_atol=0.002, median_atol=0.004, variance_rtol=0.01 if dtype == dtypes.bfloat16 else 0.005)

    @test_util.disable_mlir_bridge('b/180412086: MLIR bridge gives wrong error messages.')
    def testErrors(self):
        if False:
            return 10
        'Tests that proper errors are raised.\n    '
        shape = [2, 3]
        with ops.device(xla_device_name()):
            gen = random.Generator.from_seed(seed=1234, alg=random.RNG_ALG_THREEFRY)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, 'algorithm.* must be of shape \\[\\], not'):
                gen_stateful_random_ops.stateful_standard_normal_v2(gen.state.handle, [0, 0], shape)
            with self.assertRaisesWithPredicateMatch(TypeError, 'EagerTensor of dtype int64'):
                gen_stateful_random_ops.stateful_standard_normal_v2(gen.state.handle, 1.1, shape)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, 'Unsupported algorithm id'):
                gen_stateful_random_ops.stateful_standard_normal_v2(gen.state.handle, 123, shape)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, 'Unsupported algorithm id'):
                gen_stateful_random_ops.rng_read_and_skip(gen.state.handle, alg=123, delta=10)
            var = variables.Variable([0, 0], dtype=dtypes.uint32)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, 'Trying to read variable .* Expected int64 got'):
                gen_stateful_random_ops.stateful_standard_normal_v2(var.handle, random.RNG_ALG_THREEFRY, shape)
            var = variables.Variable([[0]], dtype=dtypes.int64)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, 'RNG state must have one and only one dimension, not'):
                gen_stateful_random_ops.stateful_standard_normal_v2(var.handle, random.RNG_ALG_THREEFRY, shape)
            var = variables.Variable([0], dtype=dtypes.int64)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, 'The size of the state must be at least'):
                gen_stateful_random_ops.stateful_standard_normal_v2(var.handle, random.RNG_ALG_THREEFRY, shape)
            var = variables.Variable([0, 0], dtype=dtypes.int64)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, 'The size of the state must be at least'):
                gen_stateful_random_ops.stateful_standard_normal_v2(var.handle, random.RNG_ALG_PHILOX, shape)
if __name__ == '__main__':
    ops.enable_eager_execution()
    config.set_soft_device_placement(False)
    test.main()