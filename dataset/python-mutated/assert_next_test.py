"""Tests for `tf.data.experimental.assert_next()`."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test

class AssertNextTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testAssertNext(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.from_tensors(0).apply(testing.assert_next(['Map'])).map(lambda x: x)
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        dataset = dataset.with_options(options)
        self.assertDatasetProduces(dataset, expected_output=[0])

    @combinations.generate(test_base.default_test_combinations())
    def testIgnoreVersionSuffix(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.from_tensors(0).apply(testing.assert_next(['Map', 'Batch'])).map(lambda x: x).batch(1)
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        dataset = dataset.with_options(options)
        self.assertDatasetProduces(dataset, expected_output=[[0]])

    @combinations.generate(test_base.default_test_combinations())
    def testAssertNextInvalid(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.from_tensors(0).apply(testing.assert_next(['Whoops']))
        self.assertDatasetProduces(dataset, expected_error=(errors.InvalidArgumentError, 'Asserted transformation matching Whoops'))

    @combinations.generate(test_base.default_test_combinations())
    def testAssertNextShort(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.from_tensors(0).apply(testing.assert_next(['Root', 'Whoops']))
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        dataset = dataset.with_options(options)
        self.assertDatasetProduces(dataset, expected_error=(errors.InvalidArgumentError, 'Asserted next 2 transformations but encountered only 1.'))
if __name__ == '__main__':
    test.main()