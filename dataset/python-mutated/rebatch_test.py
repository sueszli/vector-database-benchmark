"""Tests for `tf.data.Dataset.rebatch()`."""
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test

def _flat_shapes(dataset):
    if False:
        i = 10
        return i + 15
    return [ts.as_list() for ts in nest.flatten(dataset_ops.get_legacy_output_shapes(dataset))]

class RebatchTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testShapeInferenceNotAllBatchSizesEqual(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 1, 1])
        expected_shapes = [[None]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testShapeInferenceInputBatchDimDivisible(self, drop_remainder):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 2], drop_remainder=drop_remainder)
        expected_shapes = [[2]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testShapeInferenceInputBatchDimUnknown(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=False)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 2], drop_remainder=False)
        expected_shapes = [[None]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testShapeInferenceInputBatchDimUnknownWithDropRemainder(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=False)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 2], drop_remainder=True)
        expected_shapes = [[2]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testShapeInferenceInputBatchDimIndivisible(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(10).batch(5, drop_remainder=True)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 2], drop_remainder=False)
        expected_shapes = [[None]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testShapeInferenceInputBatchDimIndivisibleWithDropRemainder(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(10).batch(5, drop_remainder=True)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 2], drop_remainder=True)
        expected_shapes = [[2]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testBasic(self, drop_remainder):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 2], drop_remainder=drop_remainder)
        expected_shapes = [[2]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
        expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testPartialBatch(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(5).batch(4, drop_remainder=False)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 2], drop_remainder=False)
        expected_shapes = [[None]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
        expected_output = [[0, 1], [2, 3], [4]]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testPartialBatchWithDropRemainder(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(5).batch(4, drop_remainder=False)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 2], drop_remainder=True)
        expected_shapes = [[2]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
        expected_output = [[0, 1], [2, 3]]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testBatchSizeGreaterThanOriginal(self, drop_remainder):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(12).batch(4, drop_remainder=False)
        rebatched_dataset = dataset.rebatch(batch_size=[6], drop_remainder=drop_remainder)
        expected_output = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testEmptySplits(self, drop_remainder):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
        rebatched_dataset = dataset.rebatch(batch_size=[1, 1, 1, 1, 0], drop_remainder=drop_remainder)
        expected_shapes = [[None]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
        expected_output = [[0], [1], [2], [3], [], [4], [5], [6], [7], []]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testEmptyFirstSplits(self, drop_remainder):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
        rebatched_dataset = dataset.rebatch(batch_size=[0, 1], drop_remainder=drop_remainder)
        expected_shapes = [[None]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
        expected_output = [[], [0], [], [1], [], [2], [], [3], [], [4], [], [5], [], [6], [], [7], []]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testEmptyLastSplits(self, drop_remainder):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
        rebatched_dataset = dataset.rebatch(batch_size=[1, 0], drop_remainder=drop_remainder)
        expected_shapes = [[None]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
        expected_output = [[0], [], [1], [], [2], [], [3], [], [4], [], [5], [], [6], [], [7], []]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testScalarBatchSizeInput(self, drop_remainder):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
        rebatched_dataset = dataset.rebatch(batch_size=2, drop_remainder=drop_remainder)
        expected_shapes = [[2]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
        expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testMultipleBatches(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(16).batch(2, drop_remainder=True).batch(4, drop_remainder=True)
        self.assertEqual([[4, 2]], _flat_shapes(dataset))
        rebatched_dataset = dataset.rebatch([2, 2])
        self.assertEqual([[2, 2]], _flat_shapes(rebatched_dataset))
        expected_output = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testNestedDictionaryOutput(self):
        if False:
            return 10

        def map_fn(x):
            if False:
                print('Hello World!')
            return {'a': x, 'b': {'c': x + 1}}
        dataset = dataset_ops.Dataset.range(8).map(map_fn).batch(4, drop_remainder=True)
        rebatched_dataset = dataset.rebatch([2, 2])
        self.assertEqual([[2], [2]], _flat_shapes(rebatched_dataset))
        expected_output = [{'a': [0, 1], 'b': {'c': [1, 2]}}, {'a': [2, 3], 'b': {'c': [3, 4]}}, {'a': [4, 5], 'b': {'c': [5, 6]}}, {'a': [6, 7], 'b': {'c': [7, 8]}}]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testRaggedDataset(self, drop_remainder):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.from_tensor_slices(ragged_tensor.RaggedTensor.from_row_lengths(list(range(10)), [1, 2, 3, 4]))
        dataset = dataset.batch(4, drop_remainder=True).map(lambda x: x)
        rebatched_dataset = dataset.rebatch(batch_size=[2, 2])
        expected_output = [ragged_tensor.RaggedTensor.from_row_lengths(list(range(3)), [1, 2]), ragged_tensor.RaggedTensor.from_row_lengths(list(range(3, 10)), [3, 4])]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testNoneDataset(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(4)
        dataset = dataset.map(lambda x: (x, None))
        dataset = dataset.batch(4, drop_remainder=True)
        _ = dataset.rebatch(batch_size=[2, 2])

class RebatchDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations()))
    def test(self, verify_fn):
        if False:
            while True:
                i = 10

        def build_dataset(num_elements, batch_size):
            if False:
                while True:
                    i = 10
            dataset = dataset_ops.Dataset.range(num_elements)
            dataset_batched = dataset.batch(2 * batch_size, drop_remainder=True)
            return dataset_batched.rebatch(batch_size=[batch_size, batch_size])
        verify_fn(self, lambda : build_dataset(64, 8), num_outputs=8)
if __name__ == '__main__':
    test.main()