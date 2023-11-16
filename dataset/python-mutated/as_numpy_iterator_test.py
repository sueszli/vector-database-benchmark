"""Tests for `tf.data.Dataset.numpy()`."""
import collections
import os
from absl.testing import parameterized
import numpy as np
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import test
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops

class AsNumpyIteratorTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.eager_only_combinations())
    def testBasic(self):
        if False:
            return 10
        ds = dataset_ops.Dataset.range(3)
        self.assertEqual([0, 1, 2], list(ds.as_numpy_iterator()))

    @combinations.generate(test_base.eager_only_combinations())
    def testImmutable(self):
        if False:
            i = 10
            return i + 15
        ds = dataset_ops.Dataset.from_tensors([1, 2, 3])
        arr = next(ds.as_numpy_iterator())
        with self.assertRaisesRegex(ValueError, 'assignment destination is read-only'):
            arr[0] = 0

    @combinations.generate(test_base.eager_only_combinations())
    def testNestedStructure(self):
        if False:
            for i in range(10):
                print('nop')
        point = collections.namedtuple('Point', ['x', 'y'])
        ds = dataset_ops.Dataset.from_tensor_slices({'a': ([1, 2], [3, 4]), 'b': [5, 6], 'c': point([7, 8], [9, 10])})
        self.assertEqual([{'a': (1, 3), 'b': 5, 'c': point(7, 9)}, {'a': (2, 4), 'b': 6, 'c': point(8, 10)}], list(ds.as_numpy_iterator()))

    @combinations.generate(test_base.graph_only_combinations())
    def testNonEager(self):
        if False:
            i = 10
            return i + 15
        ds = dataset_ops.Dataset.range(10)
        with self.assertRaises(RuntimeError):
            ds.as_numpy_iterator()

    def _testInvalidElement(self, element):
        if False:
            for i in range(10):
                print('nop')
        ds = dataset_ops.Dataset.from_tensors(element)
        with self.assertRaisesRegex(TypeError, 'is not supported for datasets that'):
            ds.as_numpy_iterator()

    @combinations.generate(test_base.eager_only_combinations())
    def testSparseElement(self):
        if False:
            for i in range(10):
                print('nop')
        st = sparse_tensor.SparseTensor(indices=[(0, 0), (1, 1), (2, 2)], values=[1, 2, 3], dense_shape=(3, 3))
        ds = dataset_ops.Dataset.from_tensor_slices(st)
        dt = sparse_ops.sparse_tensor_to_dense(st)
        self.assertAllEqual(list(ds.as_numpy_iterator()), dt.numpy())

    @combinations.generate(test_base.eager_only_combinations())
    def testRaggedElement(self):
        if False:
            return 10
        lst = [[1, 2], [3], [4, 5, 6]]
        rt = ragged_factory_ops.constant([lst])
        ds = dataset_ops.Dataset.from_tensor_slices(rt)
        expected = np.array([np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32)], dtype=object)
        for actual in ds.as_numpy_iterator():
            self.assertEqual(len(actual), len(expected))
            for (actual_arr, expected_arr) in zip(actual, expected):
                self.assertTrue(np.array_equal(actual_arr, expected_arr), f'{actual_arr} != {expected_arr}')

    @combinations.generate(test_base.eager_only_combinations())
    def testDatasetElement(self):
        if False:
            while True:
                i = 10
        self._testInvalidElement(dataset_ops.Dataset.range(3))

    @combinations.generate(test_base.eager_only_combinations())
    def testNestedNonTensorElement(self):
        if False:
            for i in range(10):
                print('nop')
        tuple_elem = (constant_op.constant([1, 2, 3]), dataset_ops.Dataset.range(3))
        self._testInvalidElement(tuple_elem)

    @combinations.generate(test_base.eager_only_combinations())
    def testNoneElement(self):
        if False:
            i = 10
            return i + 15
        ds = dataset_ops.Dataset.from_tensors((2, None))
        self.assertDatasetProduces(ds, [(2, None)])

    @combinations.generate(combinations.times(test_base.eager_only_combinations(), combinations.combine(enable_async_ckpt=[True, False])))
    def testCompatibleWithCheckpoint(self, enable_async_ckpt):
        if False:
            return 10
        ds = dataset_ops.Dataset.range(10)
        iterator = ds.as_numpy_iterator()
        ckpt = trackable_utils.Checkpoint(iterator=iterator)
        ckpt_options = checkpoint_options.CheckpointOptions(experimental_enable_async_checkpoint=enable_async_ckpt)
        for _ in range(5):
            next(iterator)
        prefix = os.path.join(self.get_temp_dir(), 'ckpt')
        save_path = ckpt.save(prefix, options=ckpt_options)
        self.assertEqual(5, next(iterator))
        self.assertEqual(6, next(iterator))
        restore_iter = ds.as_numpy_iterator()
        restore_ckpt = trackable_utils.Checkpoint(iterator=restore_iter)
        if enable_async_ckpt:
            ckpt.sync()
        restore_ckpt.restore(save_path)
        self.assertEqual(5, next(restore_iter))
if __name__ == '__main__':
    test.main()