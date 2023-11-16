"""Tests for `tf.data.experimental.assert_prev()`."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test

class AssertPrevTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testAssertPrev(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.from_tensors(0).map(lambda x: x, deterministic=True, num_parallel_calls=8).apply(testing.assert_prev([('ParallelMapDataset', {'deterministic', 'true'})]))
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        dataset = dataset.with_options(options)
        self.assertDatasetProduces(dataset, expected_output=[0])

    @combinations.generate(test_base.default_test_combinations())
    def testIgnoreVersionSuffix(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.from_tensors(0).map(lambda x: x, deterministic=True, num_parallel_calls=8).batch(1).apply(testing.assert_prev([('BatchDataset', {}), ('ParallelMapDataset', {'deterministic': 'true'})]))
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        dataset = dataset.with_options(options)
        self.assertDatasetProduces(dataset, expected_output=[[0]])

    @combinations.generate(test_base.default_test_combinations())
    def testAssertPrevInvalid(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.from_tensors(0).apply(testing.assert_prev([('Whoops', {})]))
        self.assertDatasetProduces(dataset, expected_error=(errors.InvalidArgumentError, "Asserted transformation matching 'Whoops'"))

    @combinations.generate(test_base.default_test_combinations())
    def testAssertPrevShort(self):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.from_tensors(0).apply(testing.assert_prev([('TensorDataset', {}), ('Whoops', {})]))
        self.assertDatasetProduces(dataset, expected_error=(errors.InvalidArgumentError, 'Asserted previous 2 transformations but encountered only 1.'))

    @combinations.generate(test_base.default_test_combinations())
    def testAssertBadAttributeName(self):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.from_tensors(0).apply(testing.assert_prev([('TensorDataset', {'whoops': 'true'})]))
        self.assertDatasetProduces(dataset, expected_error=(errors.InvalidArgumentError, 'found no such attribute'))
if __name__ == '__main__':
    test.main()