"""Tests for `tf.data.Dataset.from_sparse_tensor_slices()`."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class FromSparseTensorSlicesTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(combinations.combine(tf_api_version=1, mode=['graph']), combinations.combine(slices=[[[1.0, 2.0, 3.0], [1.0], [1.0], [1.0, 2.0], [], [1.0, 2.0], [], [], []], [[1.0, 2.0], [], [1.0, 2.0], [1.0], [1.0, 2.0], [], [1.0, 2.0]]])))
    def testFromSparseTensorSlices(self, slices):
        if False:
            while True:
                i = 10
        'Test a dataset based on slices of a `tf.sparse.SparseTensor`.'
        st = array_ops.sparse_placeholder(dtypes.float64)
        iterator = dataset_ops.make_initializable_iterator(dataset_ops.Dataset.from_sparse_tensor_slices(st))
        init_op = iterator.initializer
        get_next = sparse_tensor.SparseTensor(*iterator.get_next())
        with self.cached_session() as sess:
            indices = np.array([[i, j] for i in range(len(slices)) for j in range(len(slices[i]))])
            values = np.array([val for s in slices for val in s])
            dense_shape = np.array([len(slices), max((len(s) for s in slices)) + 1])
            sparse_feed = sparse_tensor.SparseTensorValue(indices, values, dense_shape)
            sess.run(init_op, feed_dict={st: sparse_feed})
            for (i, s) in enumerate(slices):
                results = sess.run(get_next)
                self.assertAllEqual(s, results.values)
                expected_indices = np.array([[j] for j in range(len(slices[i]))]).reshape([-1, 1])
                self.assertAllEqual(expected_indices, results.indices)
                self.assertAllEqual(dense_shape[1:], results.dense_shape)
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

    @combinations.generate(combinations.times(combinations.combine(tf_api_version=1, mode=['graph']), combinations.combine(slices=[[[1.0, 2.0, 3.0], [1.0], [1.0], [1.0, 2.0], [], [1.0, 2.0], [], [], []], [[1.0, 2.0], [], [1.0, 2.0], [1.0], [1.0, 2.0], [], [1.0, 2.0]]])))
    def testFromSparseTensorSlicesInReverse(self, slices):
        if False:
            for i in range(10):
                print('nop')
        'Test a dataset based on slices of a `tf.sparse.SparseTensor` in reverse order.'
        st = array_ops.sparse_placeholder(dtypes.float64)
        iterator = dataset_ops.make_initializable_iterator(dataset_ops.Dataset.from_sparse_tensor_slices(st))
        init_op = iterator.initializer
        with self.cached_session() as sess:
            indices = np.array([[i, j] for i in range(len(slices)) for j in range(len(slices[i]))])
            values = np.array([val for s in slices for val in s])
            dense_shape = np.array([len(slices), max((len(s) for s in slices)) + 1])
            reverse_order_indices = indices[::-1, :]
            reverse_order_values = values[::-1]
            sparse_feed = sparse_tensor.SparseTensorValue(reverse_order_indices, reverse_order_values, dense_shape)
            with self.assertRaises(errors.UnimplementedError):
                sess.run(init_op, feed_dict={st: sparse_feed})

    @combinations.generate(combinations.combine(tf_api_version=1, mode=['graph']))
    def testEmptySparseTensorSlices(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a dataset based on slices of an empty `tf.sparse.SparseTensor`.'
        st = array_ops.sparse_placeholder(dtypes.float64)
        iterator = dataset_ops.make_initializable_iterator(dataset_ops.Dataset.from_sparse_tensor_slices(st))
        init_op = iterator.initializer
        get_next = sparse_tensor.SparseTensor(*iterator.get_next())
        with self.cached_session() as sess:
            empty_indices = np.empty((0, 4), dtype=np.int64)
            empty_values = np.empty((0,), dtype=np.float64)
            empty_dense_shape = [0, 4, 37, 9]
            sparse_feed = sparse_tensor.SparseTensorValue(empty_indices, empty_values, empty_dense_shape)
            sess.run(init_op, feed_dict={st: sparse_feed})
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

    @combinations.generate(combinations.combine(tf_api_version=1, mode=['graph']))
    def testEmptySparseTensorSlicesInvalid(self):
        if False:
            while True:
                i = 10
        'Test a dataset based on invalid `tf.sparse.SparseTensor`.'
        st = array_ops.sparse_placeholder(dtypes.float64)
        iterator = dataset_ops.make_initializable_iterator(dataset_ops.Dataset.from_sparse_tensor_slices(st))
        init_op = iterator.initializer
        with self.cached_session() as sess:
            empty_indices = np.empty((0, 4), dtype=np.int64)
            non_empty_values = [1, 2, 3, 4]
            empty_dense_shape = [0, 4, 37, 9]
            sparse_feed = sparse_tensor.SparseTensorValue(empty_indices, non_empty_values, empty_dense_shape)
            with self.assertRaises(errors.InvalidArgumentError):
                sess.run(init_op, feed_dict={st: sparse_feed})

    @combinations.generate(combinations.combine(tf_api_version=1, mode=['graph']))
    def testEmptySparseTensorSlicesInvalid2(self):
        if False:
            return 10
        'Test a dataset based on invalid `tf.sparse.SparseTensor`.'
        st = array_ops.sparse_placeholder(dtypes.float64)
        iterator = dataset_ops.make_initializable_iterator(dataset_ops.Dataset.from_sparse_tensor_slices(st))
        init_op = iterator.initializer
        with self.cached_session() as sess:
            empty_indices = [[]]
            empty_values = []
            dense_shape = [1, 1]
            sparse_feed = sparse_tensor.SparseTensorValue(empty_indices, empty_values, dense_shape)
            with self.assertRaises(errors.InvalidArgumentError):
                sess.run(init_op, feed_dict={st: sparse_feed})

    @combinations.generate(combinations.combine(tf_api_version=2, mode=['eager']))
    def testFromSparseTensorSlicesError(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AttributeError):
            dataset_ops.Dataset.from_sparse_tensor_slices(None)

class FromSparseTensorSlicesCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_sparse_tensor_slice_dataset(self, slices):
        if False:
            return 10
        indices = np.array([[i, j] for i in range(len(slices)) for j in range(len(slices[i]))], dtype=np.int64)
        values = np.array([val for s in slices for val in s], dtype=np.float64)
        dense_shape = np.array([len(slices), max((len(s) for s in slices)) + 1], dtype=np.int64)
        sparse_components = sparse_tensor.SparseTensor(indices, values, dense_shape)
        return dataset_ops.Dataset.from_sparse_tensor_slices(sparse_components)

    @combinations.generate(combinations.times(test_base.v1_only_combinations(), checkpoint_test_base.default_test_combinations()))
    def test(self, verify_fn):
        if False:
            return 10
        slices = [[1.0, 2.0, 3.0], [1.0], [1.0], [1.0, 2.0], [], [1.0, 2.0], [], [], []]
        verify_fn(self, lambda : self._build_sparse_tensor_slice_dataset(slices), num_outputs=9, sparse_tensors=True)
if __name__ == '__main__':
    test.main()