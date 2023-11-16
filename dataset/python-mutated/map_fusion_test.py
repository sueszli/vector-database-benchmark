"""Tests for the `MapFusion` optimization."""
import functools
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

def _test_combinations():
    if False:
        while True:
            i = 10
    cases = []
    identity = lambda x: x
    increment = lambda x: x + 1

    def increment_and_square(x):
        if False:
            print('Hello World!')
        y = x + 1
        return y * y
    functions = [identity, increment, increment_and_square]
    for (i, x) in enumerate(functions):
        for (j, y) in enumerate(functions):
            cases.append(('Scalar{}{}'.format(i, j), [x, y]))
            for (k, z) in enumerate(functions):
                cases.append(('Scalar{}{}{}'.format(i, j, k), [x, y, z]))
    with_42 = lambda x: (x, 42)
    swap = lambda x, y: (y, x)
    cases.append(('Tuple1', [with_42, swap]))
    cases.append(('Tuple2', [with_42, swap, swap]))

    def reduce_fn(x, y):
        if False:
            while True:
                i = 10
        (name, functions) = y
        return x + combinations.combine(functions=combinations.NamedObject(name, functions))
    return functools.reduce(reduce_fn, cases, [])

class MapFusionTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), _test_combinations(), combinations.combine(num_parallel_calls=[None, 2, dataset_ops.AUTOTUNE]), combinations.combine(deterministic=[None, True, False])))
    def testMapFusion(self, functions, num_parallel_calls, deterministic):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(5)
        if num_parallel_calls is None:
            dataset = dataset.apply(testing.assert_next(['Map', 'MemoryCacheImpl']))
        elif num_parallel_calls in [dataset_ops.AUTOTUNE]:
            dataset = dataset.apply(testing.assert_next(['ParallelMap', 'MemoryCacheImpl']))
        else:
            dataset = dataset.apply(testing.assert_next(['ParallelMap', 'ParallelMap']))
        for function in functions:
            dataset = dataset.map(function, num_parallel_calls=num_parallel_calls, deterministic=deterministic)
        dataset = dataset.cache()
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.map_fusion = True
        dataset = dataset.with_options(options)
        expected_output = []
        for x in range(5):
            r = x
            for function in functions:
                if isinstance(r, tuple):
                    r = function(*r)
                else:
                    r = function(r)
            expected_output.append(r)
        if num_parallel_calls is None or deterministic in [None, True]:
            self.assertDatasetProduces(dataset, expected_output=expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(num_parallel_calls=[None, 2, dataset_ops.AUTOTUNE])))
    def testCapturedInputs(self, num_parallel_calls):
        if False:
            while True:
                i = 10
        a = constant_op.constant(3, dtype=dtypes.int64)
        b = constant_op.constant(4, dtype=dtypes.int64)
        some_tensor = math_ops.mul(a, b)
        dataset = dataset_ops.Dataset.range(1)
        if num_parallel_calls in [2, dataset_ops.AUTOTUNE]:
            dataset = dataset.apply(testing.assert_next(['ParallelMap', 'ParallelMap']))
        else:
            dataset = dataset.apply(testing.assert_next(['Map', 'Map']))
        dataset = dataset.map(lambda x: some_tensor, num_parallel_calls=num_parallel_calls).map(lambda x: x, num_parallel_calls=num_parallel_calls)
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.map_fusion = True
        dataset = dataset.with_options(options)
        self.assertDatasetProduces(dataset, expected_output=[some_tensor])
if __name__ == '__main__':
    test.main()