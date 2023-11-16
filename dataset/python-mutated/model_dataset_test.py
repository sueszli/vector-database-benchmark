"""Tests for the private `_ModelDataset` transformation."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import map_op
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test

class ModelDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testAutotuneOption(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.from_tensors(0)
        dataset = dataset.map(lambda x: x).apply(testing.assert_next(['Root']))
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.autotune.enabled = True
        dataset = dataset.with_options(options)
        get_next = self.getNext(dataset)
        self.assertEqual(0, self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testParallelMapWithAutotune(self):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.range(1000)
        dataset = map_op._ParallelMapDataset(dataset, lambda x: x + 1, num_parallel_calls=1, deterministic=True, use_inter_op_parallelism=False)
        dataset = dataset.map(lambda x: x + 1, num_parallel_calls=dataset_ops.AUTOTUNE)
        next_element = self.getNext(dataset)
        self.evaluate(next_element())
if __name__ == '__main__':
    test.main()