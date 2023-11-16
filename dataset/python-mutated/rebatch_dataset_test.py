"""Tests for the private `_RebatchDataset` transformation."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import image_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test

class BatchSizesForWorkerTest(test_base.DatasetTestBase, parameterized.TestCase):

    def _test(self, global_batch_size, num_workers, num_replicas_per_worker, is_batch_size_static):
        if False:
            for i in range(10):
                print('nop')
        'Test that all constraints are met for given parameters.'
        if not is_batch_size_static:
            global_batch_size += constant_op.constant(0, dtypes.int64)
        batch_sizes_list = []
        for i in range(num_workers):
            batch_sizes_list.append(self.evaluate(distribute.batch_sizes_for_worker(global_batch_size, num_workers, num_replicas_per_worker, i)))
        for batch_sizes in batch_sizes_list:
            self.assertLen(batch_sizes, num_workers * num_replicas_per_worker)
            self.assertAllEqual(np.sum(batch_sizes), global_batch_size)
        for step_index in range(num_workers):
            actual_global_batch = 0
            offset = step_index * num_replicas_per_worker
            for batch_sizes in batch_sizes_list:
                actual_global_batch += np.sum(batch_sizes[offset:offset + num_replicas_per_worker])
            self.assertAllEqual(global_batch_size, actual_global_batch)
        self.assertLessEqual(np.max(batch_sizes_list) - np.min(batch_sizes_list), 1)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(is_batch_size_static=[True, False])))
    def testBasic(self, is_batch_size_static):
        if False:
            while True:
                i = 10
        global_batch_size = 8
        num_workers = 2
        num_replicas_per_worker = 2
        for worker_index in range(4):
            batch_sizes = distribute.batch_sizes_for_worker(global_batch_size, num_workers, num_replicas_per_worker, worker_index)
            self.assertAllEqual([2, 2, 2, 2], tensor_util.constant_value(batch_sizes))
        self._test(global_batch_size, num_workers, num_replicas_per_worker, is_batch_size_static)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(is_batch_size_static=[True, False])))
    def testBatchSizeIndivisibleByNumWorkers(self, is_batch_size_static):
        if False:
            for i in range(10):
                print('nop')
        global_batch_size = 4
        num_workers = 3
        num_replicas_per_worker = 1

        def get_batch_sizes_for_worker(worker_index):
            if False:
                for i in range(10):
                    print('nop')
            return tensor_util.constant_value(distribute.batch_sizes_for_worker(global_batch_size, num_workers, num_replicas_per_worker, worker_index))
        self.assertAllEqual([2, 1, 1], get_batch_sizes_for_worker(0))
        self.assertAllEqual([1, 1, 2], get_batch_sizes_for_worker(1))
        self.assertAllEqual([1, 2, 1], get_batch_sizes_for_worker(2))
        self._test(global_batch_size, num_workers, num_replicas_per_worker, is_batch_size_static)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(is_batch_size_static=[True, False])))
    def testBatchSizeIndivisibleByNumReplicas(self, is_batch_size_static):
        if False:
            while True:
                i = 10
        self._test(global_batch_size=4, num_workers=1, num_replicas_per_worker=5, is_batch_size_static=is_batch_size_static)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(is_batch_size_static=[True, False])))
    def testBatchSizeSmallerThanNumReplicas(self, is_batch_size_static):
        if False:
            i = 10
            return i + 15
        self._test(global_batch_size=4, num_workers=2, num_replicas_per_worker=5, is_batch_size_static=is_batch_size_static)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(is_batch_size_static=[True, False])))
    def testBatchSizeSmallerThanNumWorkers(self, is_batch_size_static):
        if False:
            i = 10
            return i + 15
        self._test(global_batch_size=4, num_workers=5, num_replicas_per_worker=1, is_batch_size_static=is_batch_size_static)

def _flat_shapes(dataset):
    if False:
        return 10
    return [ts.as_list() for ts in nest.flatten(dataset_ops.get_legacy_output_shapes(dataset))]

class LegacyRebatchDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testBasic(self, drop_remainder):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=drop_remainder)
        rebatched_dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=2)
        expected_shapes = [[2]] if drop_remainder else [[None]]
        self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
        expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testCanHandleUnknownRank(self):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.from_tensors('xxx')
        dataset = dataset.map(image_ops.decode_image)
        self.assertEqual([tensor_shape.TensorShape(None)], nest.flatten(dataset_ops.get_legacy_output_shapes(dataset)))
        rebatched_dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=4)
        self.assertEqual([tensor_shape.TensorShape(None)], nest.flatten(dataset_ops.get_legacy_output_shapes(rebatched_dataset)))

    @combinations.generate(test_base.default_test_combinations())
    def testCanHandleUnknownDims(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(1000)
        dataset = dataset.batch(10, drop_remainder=False)
        dataset = dataset.batch(10, drop_remainder=False)
        self.assertEqual([[None, None]], _flat_shapes(dataset))
        rebatched_dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=4)
        self.assertEqual([[None, None]], _flat_shapes(rebatched_dataset))

    @combinations.generate(test_base.default_test_combinations())
    def testScalarInputError(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(1024)
        distribute._LegacyRebatchDataset(dataset.batch(4), num_replicas=4)
        with self.assertRaises(ValueError):
            distribute._LegacyRebatchDataset(dataset, num_replicas=4)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testBatchNotDivisibleByNumReplicas(self, drop_remainder):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=drop_remainder)
        rebatched_dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=3)
        self.assertEqual([[None]], _flat_shapes(rebatched_dataset))
        expected_output = [[0, 1], [2, 3], [], [4, 5], [6, 7], []]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testTupleOutput(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(1024).map(lambda x: (x, x)).batch(32)
        rebatched_dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=4)
        expected_output = [([k for k in range(i, i + 8)], [k for k in range(i, i + 8)]) for i in range(0, 1024, 8)]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testNestedDictionaryOutput(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(8).map(lambda x: {'a': x, 'b': {'c': x + 1}}).batch(4)
        rebatched_dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=2)
        expected_output = [{'a': [0, 1], 'b': {'c': [1, 2]}}, {'a': [2, 3], 'b': {'c': [3, 4]}}, {'a': [4, 5], 'b': {'c': [5, 6]}}, {'a': [6, 7], 'b': {'c': [7, 8]}}]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testFinalPartialBatch(self, drop_remainder):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(10).batch(4, drop_remainder=drop_remainder)
        rebatched_dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=2)
        self.assertEqual([[2] if drop_remainder else [None]], _flat_shapes(rebatched_dataset))
        if drop_remainder:
            expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
        else:
            expected_output = [[0, 1], [2, 3], [4, 5], [6, 7], [8], [9]]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(drop_remainder=[True, False])))
    def testFinalPartialBatchAfterRebatch(self, drop_remainder):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(9).batch(4, drop_remainder=drop_remainder)
        rebatched_dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=2)
        self.assertEqual([[2] if drop_remainder else [None]], _flat_shapes(rebatched_dataset))
        if drop_remainder:
            expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
        else:
            expected_output = [[0, 1], [2, 3], [4, 5], [6, 7], [8], []]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testMultipleBatches(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(16).batch(2).batch(4)
        self.assertEqual([[None, None]], _flat_shapes(dataset))
        expected_output = [[[0, 1], [2, 3], [4, 5], [6, 7]], [[8, 9], [10, 11], [12, 13], [14, 15]]]
        self.assertDatasetProduces(dataset, expected_output)
        rebatched_dataset = distribute._LegacyRebatchDataset(dataset, 2)
        self.assertEqual([[None, None]], _flat_shapes(rebatched_dataset))
        expected_output = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]]
        self.assertDatasetProduces(rebatched_dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testRaggedTensorDataset(self):
        if False:
            while True:
                i = 10
        row_lengths = np.random.randint(8, size=128)
        values = np.random.normal(size=np.sum(row_lengths)).astype(np.float32)
        dataset = dataset_ops.Dataset.from_tensor_slices(ragged_tensor.RaggedTensor.from_row_lengths(values, row_lengths))
        dataset = dataset.batch(32, drop_remainder=True)
        dataset = dataset.map(lambda x: x)
        dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=8)
        expected_output = []
        value_index = 0
        for batch_row_lengths in row_lengths.reshape((-1, 4)):
            num_values = np.sum(batch_row_lengths)
            expected_output.append(ragged_tensor.RaggedTensor.from_row_lengths(values[value_index:value_index + num_values], batch_row_lengths))
            value_index += num_values
        self.assertDatasetProduces(dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testNoneDataset(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(4)
        dataset = dataset.map(lambda x: (x, None))
        dataset = dataset.batch(4, drop_remainder=True)
        _ = distribute._LegacyRebatchDataset(dataset, num_replicas=2)

class ComputeBatchSizeTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testComputeBatchSizeKnown(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(32).batch(4, drop_remainder=True)
        dataset = dataset_ops.Dataset.zip((dataset, dataset))
        batch_size = distribute.compute_batch_size(dataset)
        self.assertEqual(4, self.evaluate(batch_size))

    @combinations.generate(test_base.default_test_combinations())
    def testComputeBatchSizeKnownAndMismatched(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(32)
        dataset = dataset_ops.Dataset.zip((dataset.batch(4, drop_remainder=True), dataset.batch(8, drop_remainder=True)))
        batch_size = distribute.compute_batch_size(dataset)
        self.assertEqual(-1, self.evaluate(batch_size))

    @combinations.generate(test_base.default_test_combinations())
    def testComputeBatchSizeUnknown(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(32).batch(4)
        batch_size = distribute.compute_batch_size(dataset)
        self.assertEqual(4, self.evaluate(batch_size))

    @combinations.generate(test_base.default_test_combinations())
    def testComputeBatchSizeWithPassthrough(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(32).batch(4)
        dataset = dataset.take(5)
        batch_size = distribute.compute_batch_size(dataset)
        self.assertEqual(4, self.evaluate(batch_size))

    @combinations.generate(test_base.default_test_combinations())
    def testComputeBatchSizeWithPassthroughInvalid(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(32).batch(4)
        dataset = dataset.map(lambda x: x + 1)
        batch_size = distribute.compute_batch_size(dataset)
        self.assertEqual(-1, self.evaluate(batch_size))

    @combinations.generate(test_base.default_test_combinations())
    def testComputeBatchSizeWithZip(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(32).batch(4)
        dataset = dataset_ops.Dataset.zip((dataset, dataset))
        batch_size = distribute.compute_batch_size(dataset)
        self.assertEqual(4, self.evaluate(batch_size))

    @combinations.generate(test_base.default_test_combinations())
    def testComputeBatchSizeWithZipMismatched(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(32)
        dataset = dataset_ops.Dataset.zip((dataset.batch(4), dataset.batch(8)))
        batch_size = distribute.compute_batch_size(dataset)
        self.assertEqual(-1, self.evaluate(batch_size))

    @combinations.generate(test_base.default_test_combinations())
    def testNoneDataset(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(4)
        dataset = dataset.map(lambda x: (x, None))
        dataset = dataset.batch(4, drop_remainder=True)
        batch_size = distribute.compute_batch_size(dataset)
        self.assertEqual(4, self.evaluate(batch_size))

class LegacyRebatchDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations()))
    def test(self, verify_fn):
        if False:
            return 10

        def build_dataset(num_elements, batch_size):
            if False:
                i = 10
                return i + 15
            return distribute._LegacyRebatchDataset(dataset_ops.Dataset.range(num_elements).batch(4 * batch_size, drop_remainder=True), num_replicas=4)
        verify_fn(self, lambda : build_dataset(64, 8), num_outputs=8)
if __name__ == '__main__':
    test.main()