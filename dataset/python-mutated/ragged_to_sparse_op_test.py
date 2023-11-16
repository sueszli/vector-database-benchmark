"""Tests for ragged.to_sparse op."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorToSparseOpTest(test_util.TensorFlowTestCase):

    def testDocStringExample(self):
        if False:
            print('Hello World!')
        rt = ragged_factory_ops.constant([[1, 2, 3], [4], [], [5, 6]])
        st = self.evaluate(rt.to_sparse())
        self.assertAllEqual(st.indices, [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [3, 1]])
        self.assertAllEqual(st.values, [1, 2, 3, 4, 5, 6])
        self.assertAllEqual(st.dense_shape, [4, 3])

    def test2DRaggedTensorWithOneRaggedDimension(self):
        if False:
            print('Hello World!')
        rt = ragged_factory_ops.constant([['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']])
        st = self.evaluate(rt.to_sparse())
        self.assertAllEqual(st.indices, [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0], [4, 0]])
        self.assertAllEqual(st.values, b'a b c d e f g'.split())
        self.assertAllEqual(st.dense_shape, [5, 3])

    def test3DRaggedTensorWithOneRaggedDimension(self):
        if False:
            i = 10
            return i + 15
        rt = ragged_factory_ops.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]], [[11, 12]], [], [[13, 14]]], ragged_rank=1)
        st = self.evaluate(rt.to_sparse())
        self.assertAllEqual(st.indices, [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1], [2, 0, 0], [2, 0, 1], [4, 0, 0], [4, 0, 1]])
        self.assertAllEqual(st.values, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        self.assertAllEqual(st.dense_shape, [5, 3, 2])

    def test4DRaggedTensorWithOneRaggedDimension(self):
        if False:
            while True:
                i = 10
        rt = ragged_factory_ops.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [], [[[9, 10], [11, 12]]]], ragged_rank=1)
        st = self.evaluate(rt.to_sparse())
        self.assertAllEqual(st.values, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.assertAllEqual(st.indices, [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [2, 0, 0, 0], [2, 0, 0, 1], [2, 0, 1, 0], [2, 0, 1, 1]])
        self.assertAllEqual(st.dense_shape, [3, 2, 2, 2])

    def test4DRaggedTensorWithTwoRaggedDimensions(self):
        if False:
            print('Hello World!')
        rt = ragged_factory_ops.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]]], [[[11, 12]], [], [[13, 14]]], []], ragged_rank=2)
        st = self.evaluate(rt.to_sparse())
        self.assertAllEqual(st.indices, [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 2, 0], [0, 1, 2, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 2, 0, 0], [1, 2, 0, 1]])
        self.assertAllEqual(st.values, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        self.assertAllEqual(st.dense_shape, [3, 3, 3, 2])

    def testShape(self):
        if False:
            for i in range(10):
                print('nop')
        rt = ragged_factory_ops.constant([[1, 2], [3, 4, 5], [6], [], [7]])
        st = rt.to_sparse()
        self.assertEqual(st.indices.shape.as_list(), [7, 2])
        self.assertEqual(st.values.shape.as_list(), [7])
        self.assertEqual(st.dense_shape.shape.as_list(), [2])
        rt = ragged_factory_ops.constant([[[1, 2]], [], [[3, 4]], []], ragged_rank=1)
        st = rt.to_sparse()
        self.assertEqual(st.indices.shape.as_list(), [4, 3])
        self.assertEqual(st.values.shape.as_list(), [4])
        self.assertEqual(st.dense_shape.shape.as_list(), [3])
        rt = ragged_factory_ops.constant([[[1], [2, 3, 4, 5, 6, 7]], [[]]])
        st = rt.to_sparse()
        self.assertEqual(st.indices.shape.as_list(), [7, 3])
        self.assertEqual(st.values.shape.as_list(), [7])
        self.assertEqual(st.dense_shape.shape.as_list(), [3])

    def testKernelErrors(self):
        if False:
            i = 10
            return i + 15
        empty_vector = array_ops.placeholder_with_default(array_ops.zeros([0], dtypes.int64), shape=None)
        bad_rt1 = ragged_tensor.RaggedTensor.from_row_splits(row_splits=[2, 3], values=[1, 2, 3], validate=False)
        bad_split0 = 'First value of ragged splits must be 0.*'
        with self.assertRaisesRegex(errors.InvalidArgumentError, bad_split0):
            self.evaluate(bad_rt1.to_sparse())
        bad_rt2 = ragged_tensor.RaggedTensor.from_row_splits(row_splits=[0, 5], values=empty_vector, validate=False)
        bad_rt3 = ragged_tensor.RaggedTensor.from_row_splits(row_splits=[0, 1], values=ragged_tensor.RaggedTensor.from_row_splits(row_splits=[0, 5], values=empty_vector, validate=False), validate=False)
        split_mismatch1_error = 'Final value of ragged splits must match.*'
        for rt in [bad_rt2, bad_rt3]:
            with self.assertRaisesRegex(errors.InvalidArgumentError, split_mismatch1_error):
                self.evaluate(rt.to_sparse())
        bad_rt4 = ragged_tensor.RaggedTensor.from_row_splits(row_splits=[0, 5], values=ragged_tensor.RaggedTensor.from_row_splits(row_splits=[0], values=empty_vector, validate=False), validate=False)
        split_mismatch2_error = 'Final value of ragged splits must match.*'
        with self.assertRaisesRegex(errors.InvalidArgumentError, split_mismatch2_error):
            self.evaluate(bad_rt4.to_sparse())
        bad_rt5 = ragged_tensor.RaggedTensor.from_row_splits(row_splits=empty_vector, values=[], validate=False)
        empty_splits_error = 'ragged splits may not be empty.*'
        with self.assertRaisesRegex(errors.InvalidArgumentError, empty_splits_error):
            self.evaluate(bad_rt5.to_sparse())

    def testGradient(self):
        if False:
            print('Hello World!')
        if context.executing_eagerly():
            return
        rt1 = ragged_factory_ops.constant([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]], ragged_rank=2)
        rt2 = ragged_factory_ops.constant([[[[9.0, 8.0], [7.0, 6.0]], [[5.0, 4.0]]]], ragged_rank=2)
        rt = ragged_functional_ops.map_flat_values(math_ops.add, rt1, rt2 * 2.0)
        st = rt.to_sparse()
        (g1, g2) = gradients_impl.gradients(st.values, [rt1.flat_values, rt2.flat_values])
        self.assertAllEqual(g1, [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        self.assertAllEqual(g2, [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
if __name__ == '__main__':
    googletest.main()