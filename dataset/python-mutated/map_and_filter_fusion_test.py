"""Tests for the `MapAndFilterFusion` optimization."""
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
        print('Hello World!')
    cases = []
    identity = lambda x: x
    increment = lambda x: x + 1
    minus_five = lambda x: x - 5

    def increment_and_square(x):
        if False:
            i = 10
            return i + 15
        y = x + 1
        return y * y
    functions = [identity, increment, minus_five, increment_and_square]
    take_all = lambda x: constant_op.constant(True)
    is_zero = lambda x: math_ops.equal(x, 0)
    is_odd = lambda x: math_ops.equal(x % 2, 0)
    greater = lambda x: math_ops.greater(x + 5, 0)
    predicates = [take_all, is_zero, is_odd, greater]
    for (i, function) in enumerate(functions):
        for (j, predicate) in enumerate(predicates):
            cases.append((function, 'Scalar{}{}'.format(i, j), predicate))
    replicate = lambda x: (x, x)
    with_two = lambda x: (x, 2)
    functions = [replicate, with_two]
    take_all = lambda x, y: constant_op.constant(True)
    is_zero = lambda x, y: math_ops.equal(x * math_ops.cast(y, dtypes.int64), 0)
    predicates = [take_all, is_zero]
    for (i, function) in enumerate(functions):
        for (j, predicate) in enumerate(predicates):
            cases.append((function, 'Tuple{}{}'.format(i, j), predicate))

    def reduce_fn(x, y):
        if False:
            i = 10
            return i + 15
        (function, name, predicate) = y
        return x + combinations.combine(function=function, predicate=combinations.NamedObject(name, predicate))
    return functools.reduce(reduce_fn, cases, [])

class MapAndFilterFusionTest(test_base.DatasetTestBase, parameterized.TestCase):

    def _testDataset(self, dataset, function, predicate):
        if False:
            while True:
                i = 10
        expected_output = []
        for x in range(10):
            r = function(x)
            if isinstance(r, tuple):
                b = predicate(*r)
            else:
                b = predicate(r)
            if self.evaluate(b):
                expected_output.append(r)
        self.assertDatasetProduces(dataset, expected_output=expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), _test_combinations()))
    def testMapAndFilterFusion(self, function, predicate):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(10).apply(testing.assert_next(['Map', 'Filter', 'Map'])).map(function).filter(predicate)
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.map_and_filter_fusion = True
        dataset = dataset.with_options(options)
        self._testDataset(dataset, function, predicate)

    @combinations.generate(test_base.default_test_combinations())
    def testCapturedInputs(self):
        if False:
            for i in range(10):
                print('nop')
        a = constant_op.constant(3, dtype=dtypes.int64)
        b = constant_op.constant(4, dtype=dtypes.int64)
        some_tensor = math_ops.mul(a, b)
        function = lambda x: x * x

        def predicate(y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.less(math_ops.cast(y, dtypes.int64), some_tensor)
        dataset = dataset_ops.Dataset.range(10).apply(testing.assert_next(['Map', 'Filter'])).map(function).filter(predicate)
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.map_and_filter_fusion = True
        dataset = dataset.with_options(options)
        self._testDataset(dataset, function, predicate)
if __name__ == '__main__':
    test.main()