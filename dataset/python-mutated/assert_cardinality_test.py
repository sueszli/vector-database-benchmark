"""Tests for `tf.data.experimental.assert_cardinality()`."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test

class AssertCardinalityTest(test_base.DatasetTestBase, parameterized.TestCase):
    """Tests for `tf.data.experimental.assert_cardinality()`."""

    @combinations.generate(test_base.default_test_combinations())
    def testCorrectCardinality(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(10).filter(lambda x: True)
        self.assertEqual(self.evaluate(cardinality.cardinality(dataset)), cardinality.UNKNOWN)
        self.assertDatasetProduces(dataset, expected_output=range(10))
        dataset = dataset.apply(cardinality.assert_cardinality(10))
        self.assertEqual(self.evaluate(cardinality.cardinality(dataset)), 10)
        self.assertDatasetProduces(dataset, expected_output=range(10))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(num_elements=10, asserted_cardinality=20, expected_error='Input dataset was expected to contain 20 elements but contained only 10 elements.') + combinations.combine(num_elements=1, asserted_cardinality=20, expected_error='Input dataset was expected to contain 20 elements but contained only 1 element.') + combinations.combine(num_elements=10, asserted_cardinality=cardinality.INFINITE, expected_error='Input dataset was expected to contain an infinite number of elements but contained only 10 elements.') + combinations.combine(num_elements=1, asserted_cardinality=cardinality.INFINITE, expected_error='Input dataset was expected to contain an infinite number of elements but contained only 1 element.') + combinations.combine(num_elements=10, asserted_cardinality=5, expected_error='Input dataset was expected to contain 5 elements but contained at least 6 elements.') + combinations.combine(num_elements=10, asserted_cardinality=1, expected_error='Input dataset was expected to contain 1 element but contained at least 2 elements.')))
    def testIncorrectCardinality(self, num_elements, asserted_cardinality, expected_error):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.range(num_elements)
        dataset = dataset.apply(cardinality.assert_cardinality(asserted_cardinality))
        get_next = self.getNext(dataset)
        with self.assertRaisesRegex(errors.FailedPreconditionError, expected_error):
            while True:
                self.evaluate(get_next())

class AssertCardinalityCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def build_dataset(self, num_elements, options=None):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(num_elements).apply(cardinality.assert_cardinality(num_elements))
        if options:
            dataset = dataset.with_options(options)
        return dataset

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations(), combinations.combine(symbolic_checkpoint=[False, True])))
    def test(self, verify_fn, symbolic_checkpoint):
        if False:
            for i in range(10):
                print('nop')
        options = options_lib.Options()
        options.experimental_symbolic_checkpoint = symbolic_checkpoint
        verify_fn(self, lambda : self.build_dataset(200, options), num_outputs=200)
if __name__ == '__main__':
    test.main()