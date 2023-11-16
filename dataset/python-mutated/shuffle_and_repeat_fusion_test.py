"""Tests for the `ShuffleAndRepeatFusion` optimization."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test

class ShuffleAndRepeatFusionTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testShuffleAndRepeatFusion(self):
        if False:
            for i in range(10):
                print('nop')
        expected = 'ShuffleAndRepeat'
        dataset = dataset_ops.Dataset.range(10).apply(testing.assert_next([expected])).shuffle(10).repeat(2)
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.shuffle_and_repeat_fusion = True
        dataset = dataset.with_options(options)
        get_next = self.getNext(dataset)
        for _ in range(2):
            results = []
            for _ in range(10):
                results.append(self.evaluate(get_next()))
            self.assertAllEqual([x for x in range(10)], sorted(results))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())
if __name__ == '__main__':
    test.main()