"""Tests for tf.ragged.ragged_tensor_shape."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.ops.ragged.ragged_tensor_shape import RaggedTensorDynamicShape
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorShapeTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    def assertShapeEq(self, x, y):
        if False:
            return 10
        assert isinstance(x, RaggedTensorDynamicShape)
        assert isinstance(y, RaggedTensorDynamicShape)
        self.assertLen(x.partitioned_dim_sizes, len(y.partitioned_dim_sizes))
        for (x_dims, y_dims) in zip(x.partitioned_dim_sizes, y.partitioned_dim_sizes):
            self.assertAllEqual(x_dims, y_dims)
        self.assertAllEqual(x.inner_dim_sizes, y.inner_dim_sizes)

    @parameterized.parameters([dict(value='x', expected_dim_sizes=[]), dict(value=['a', 'b', 'c'], expected_dim_sizes=[3]), dict(value=[['a', 'b', 'c'], ['d', 'e', 'f']], expected_dim_sizes=[2, 3]), dict(value=[[['a', 'b', 'c'], ['d', 'e', 'f']]], expected_dim_sizes=[1, 2, 3]), dict(value=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d', 'e']]), expected_dim_sizes=[2, [3, 2]]), dict(value=ragged_factory_ops.constant_value([[['a', 'b', 'c'], ['d', 'e']]]), expected_dim_sizes=[1, [2], [3, 2]]), dict(value=ragged_factory_ops.constant_value([[['a', 'b', 'c'], ['d', 'e', 'f']]], ragged_rank=1), expected_dim_sizes=[1, [2], 3]), dict(value=ragged_factory_ops.constant_value([[[[1], [2]], [[3], [4]]], [[[5], [6]]]], ragged_rank=1), expected_dim_sizes=[2, [2, 1], 2, 1]), dict(value=ragged_factory_ops.constant_value([[10, 20], [30]]), expected_dim_sizes=[2, [2, 1]]), dict(value=[[1, 2, 3], [4, 5, 6]], expected_dim_sizes=[2, 3]), dict(value=ragged_factory_ops.constant_value([[1, 2], [], [3, 4, 5]]), expected_dim_sizes=[3, [2, 0, 3]]), dict(value=ragged_factory_ops.constant_value([[[1, 2], [3, 4]], [[5, 6]]], ragged_rank=1), expected_dim_sizes=[2, [2, 1], 2]), dict(value=ragged_factory_ops.constant_value([[[1, 2], [3]], [[4, 5]]]), expected_dim_sizes=[2, [2, 1], [2, 1, 2]])])
    def testFromTensor(self, value, expected_dim_sizes):
        if False:
            i = 10
            return i + 15
        shape = RaggedTensorDynamicShape.from_tensor(value)
        expected = RaggedTensorDynamicShape.from_dim_sizes(expected_dim_sizes)
        self.assertShapeEq(shape, expected)

    @parameterized.parameters([dict(dim_sizes=[], rank=0, expected_dim_sizes=[]), dict(dim_sizes=[], rank=3, expected_dim_sizes=[1, 1, 1]), dict(dim_sizes=[3], rank=1, expected_dim_sizes=[3]), dict(dim_sizes=[3], rank=3, expected_dim_sizes=[1, 1, 3]), dict(dim_sizes=[2, 3], rank=3, expected_dim_sizes=[1, 2, 3]), dict(dim_sizes=[3, [3, 2, 4]], rank=2, expected_dim_sizes=[3, [3, 2, 4]]), dict(dim_sizes=[3, [3, 2, 4]], rank=4, expected_dim_sizes=[1, 1, 3, [3, 2, 4]]), dict(dim_sizes=[3, [3, 2, 4], 2, 3], rank=5, expected_dim_sizes=[1, 3, [3, 2, 4], 2, 3])])
    def testBroadcastToRank(self, dim_sizes, rank, expected_dim_sizes):
        if False:
            i = 10
            return i + 15
        shape = RaggedTensorDynamicShape.from_dim_sizes(dim_sizes)
        expected = RaggedTensorDynamicShape.from_dim_sizes(expected_dim_sizes)
        broadcasted_shape = shape.broadcast_to_rank(rank)
        self.assertShapeEq(broadcasted_shape, expected)
        self.assertEqual(broadcasted_shape.rank, rank)

    @parameterized.parameters([dict(axis=0, row_length=3, original_dim_sizes=[1, 4, 5], broadcast_dim_sizes=[3, 4, 5]), dict(axis=2, row_length=5, original_dim_sizes=[3, 4, 1], broadcast_dim_sizes=[3, 4, 5]), dict(axis=2, row_length=5, original_dim_sizes=[3, [3, 2, 8], 1], broadcast_dim_sizes=[3, [3, 2, 8], 5]), dict(axis=5, row_length=5, original_dim_sizes=[2, [2, 1], [3, 2, 8], 3, 4, 1], broadcast_dim_sizes=[2, [2, 1], [3, 2, 8], 3, 4, 5]), dict(axis=1, row_length=[2, 0, 1], original_dim_sizes=[3, 1], broadcast_dim_sizes=[3, [2, 0, 1]]), dict(axis=1, row_length=[2, 0, 1], original_dim_sizes=[3, 1, 5], broadcast_dim_sizes=[3, [2, 0, 1], 5]), dict(axis=2, row_length=[2, 0, 1, 3, 8, 2, 3, 4, 1, 8, 7, 0], original_dim_sizes=[4, 3, 1], broadcast_dim_sizes=[4, 3, [2, 0, 1, 3, 8, 2, 3, 4, 1, 8, 7, 0]]), dict(axis=2, row_length=[2, 5, 3], original_dim_sizes=[2, [2, 1], 1], broadcast_dim_sizes=[2, [2, 1], [2, 5, 3]]), dict(axis=4, row_length=list(range(18)), original_dim_sizes=[2, [2, 1], 3, 2, 1, 8], broadcast_dim_sizes=[2, [2, 1], 3, 2, list(range(18)), 8]), dict(axis=0, row_length=3, original_dim_sizes=[1, [5]], broadcast_dim_sizes=[3, [5, 5, 5]]), dict(axis=0, row_length=2, original_dim_sizes=[1, 3, [3, 0, 2]], broadcast_dim_sizes=[2, 3, [3, 0, 2, 3, 0, 2]]), dict(axis=0, row_length=3, original_dim_sizes=[1, [3], [3, 5, 2], 9, 4, 5], broadcast_dim_sizes=[3, [3, 3, 3], [3, 5, 2, 3, 5, 2, 3, 5, 2], 9, 4, 5]), dict(axis=0, row_length=2, original_dim_sizes=[1, 2, [2, 1], [3, 5, 2], 2], broadcast_dim_sizes=[2, 2, [2, 1, 2, 1], [3, 5, 2, 3, 5, 2], 2]), dict(axis=1, row_length=2, original_dim_sizes=[3, 1, [4, 0, 2], 5], broadcast_dim_sizes=[3, 2, [4, 0, 2, 4, 0, 2], 5]), dict(axis=1, row_length=1, original_dim_sizes=[2, 3, (1, 2, 3, 4, 5, 6)], broadcast_dim_sizes=[2, 3, (1, 2, 3, 4, 5, 6)]), dict(axis=1, row_length=[4, 1, 2], original_dim_sizes=[3, 1, [3, 1, 2], 5], broadcast_dim_sizes=[3, [4, 1, 2], [3, 3, 3, 3, 1, 2, 2], 5]), dict(axis=1, row_length=[2, 0, 3], original_dim_sizes=[3, 1, [3, 1, 2], [3, 1, 4, 1, 5, 9]], broadcast_dim_sizes=[3, [2, 0, 3], [3, 3, 2, 2, 2], [3, 1, 4, 3, 1, 4, 5, 9, 5, 9, 5, 9]]), dict(axis=2, row_length=[4, 1, 2], original_dim_sizes=[3, [2, 0, 1], 1, [3, 2, 1], [1, 0, 1, 0, 2, 3], 5], broadcast_dim_sizes=[3, [2, 0, 1], [4, 1, 2], [3, 3, 3, 3, 2, 1, 1], [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 3], 5]), dict(axis=0, row_length=2, original_dim_sizes=[1, 1, 2, (2, 1)], broadcast_dim_sizes=[2, 1, 2, (2, 1, 2, 1)]), dict(axis=1, row_length=(2, 1), original_dim_sizes=[2, 1, 2, (2, 1, 2, 1)], broadcast_dim_sizes=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]), dict(axis=2, row_length=2, original_dim_sizes=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)], broadcast_dim_sizes=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]), dict(axis=3, row_length=(2, 1, 2, 1, 2, 1), original_dim_sizes=[2, (2, 1), 2, 1], broadcast_dim_sizes=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)])])
    def testBroadcastDimension(self, axis, row_length, original_dim_sizes, broadcast_dim_sizes):
        if False:
            for i in range(10):
                print('nop')
        'Tests for the broadcast_dimension method.\n\n    Verifies that:\n\n    * `original.broadcast_dimension(axis, row_length) == broadcast`\n    * `broadcast.broadcast_dimension(axis, row_length) == broadcast`\n    * `broadcast.broadcast_dimension(axis, 1) == broadcast`\n\n    Args:\n      axis: The axis to broadcast\n      row_length: The slice lengths to broadcast to.\n      original_dim_sizes: The dimension sizes before broadcasting.\n        original_dim_sizes[axis] should be equal to `1` or `row_length`.\n      broadcast_dim_sizes: THe dimension sizes after broadcasting.\n    '
        original_shape = RaggedTensorDynamicShape.from_dim_sizes(original_dim_sizes)
        bcast_shape = RaggedTensorDynamicShape.from_dim_sizes(broadcast_dim_sizes)
        self.assertEqual(original_shape.rank, bcast_shape.rank)
        bcast1 = original_shape.broadcast_dimension(axis, row_length)
        bcast2 = bcast_shape.broadcast_dimension(axis, row_length)
        bcast3 = bcast_shape.broadcast_dimension(axis, 1)
        self.assertShapeEq(bcast1, bcast_shape)
        self.assertShapeEq(bcast2, bcast_shape)
        self.assertShapeEq(bcast3, bcast_shape)

    @parameterized.parameters([dict(x_dims=[], y_dims=[], expected_dims=[]), dict(x_dims=[], y_dims=[2], expected_dims=[2]), dict(x_dims=[], y_dims=[2, 3], expected_dims=[2, 3]), dict(x_dims=[], y_dims=[2, (2, 3), (5, 7, 2, 0, 9)], expected_dims=[2, (2, 3), (5, 7, 2, 0, 9)]), dict(x_dims=[3], y_dims=[4, 2, 3], expected_dims=[4, 2, 3]), dict(x_dims=[1], y_dims=[4, 2, 3], expected_dims=[4, 2, 3]), dict(x_dims=[3], y_dims=[4, 2, 1], expected_dims=[4, 2, 3]), dict(x_dims=[3], y_dims=[3, (2, 3, 1), 1], expected_dims=[3, (2, 3, 1), 3]), dict(x_dims=[1], y_dims=[3, (2, 1, 3)], expected_dims=[3, (2, 1, 3)]), dict(x_dims=[1], y_dims=[3, (2, 1, 3), 8], expected_dims=[3, (2, 1, 3), 8]), dict(x_dims=[1], y_dims=[2, (2, 3), (5, 7, 2, 0, 9)], expected_dims=[2, (2, 3), (5, 7, 2, 0, 9)]), dict(x_dims=[1, 3, (3, 0, 2), 1, 2], y_dims=[2, 1, 1, (7, 2), 1], expected_dims=[2, 3, (3, 0, 2, 3, 0, 2), (7, 7, 7, 7, 7, 2, 2, 2, 2, 2), 2]), dict(x_dims=[2, (2, 1), 2, 1], y_dims=[1, 1, 2, (2, 1)], expected_dims=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)])])
    def testBroadcastDynamicShape(self, x_dims, y_dims, expected_dims):
        if False:
            i = 10
            return i + 15
        x_shape = RaggedTensorDynamicShape.from_dim_sizes(x_dims)
        y_shape = RaggedTensorDynamicShape.from_dim_sizes(y_dims)
        expected = RaggedTensorDynamicShape.from_dim_sizes(expected_dims)
        result1 = ragged_tensor_shape.broadcast_dynamic_shape(x_shape, y_shape)
        result2 = ragged_tensor_shape.broadcast_dynamic_shape(y_shape, x_shape)
        self.assertShapeEq(expected, result1)
        self.assertShapeEq(expected, result2)

    def testRepr(self):
        if False:
            return 10
        shape = RaggedTensorDynamicShape.from_dim_sizes([2, (2, 1), 2, 1])
        self.assertRegex(repr(shape), 'RaggedTensorDynamicShape\\(partitioned_dim_sizes=\\(<[^>]+>, <[^>]+>\\), inner_dim_sizes=<[^>]+>\\)')

    @parameterized.parameters([dict(x=[[10], [20], [30]], dim_sizes=[3, 2], expected=[[10, 10], [20, 20], [30, 30]]), dict(x=[[10], [20], [30]], dim_sizes=[3, [3, 0, 2]], expected=ragged_factory_ops.constant_value([[10, 10, 10], [], [30, 30]], dtype=np.int32)), dict(x=[[[1, 2, 3]], [[4, 5, 6]]], dim_sizes=[2, [2, 3], 3], expected=ragged_factory_ops.constant_value([[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6]]], dtype=np.int32, ragged_rank=1)), dict(x=[[[1]], [[2]]], dim_sizes=[2, [2, 3], [0, 2, 1, 2, 0]], expected=ragged_factory_ops.constant_value([[[], [1, 1]], [[2], [2, 2], []]], dtype=np.int32, ragged_rank=2)), dict(x=10, dim_sizes=[3, [3, 0, 2]], expected=ragged_factory_ops.constant_value([[10, 10, 10], [], [10, 10]])), dict(x=ragged_factory_ops.constant_value([[[1], [2]], [[3]]], ragged_rank=1), dim_sizes=[2, [2, 1], 2], expected=ragged_factory_ops.constant_value([[[1, 1], [2, 2]], [[3, 3]]], ragged_rank=1))])
    def testRaggedBroadcastTo(self, x, dim_sizes, expected):
        if False:
            return 10
        shape = RaggedTensorDynamicShape.from_dim_sizes(dim_sizes)
        result = ragged_tensor_shape.broadcast_to(x, shape)
        self.assertEqual(getattr(result, 'ragged_rank', 0), getattr(expected, 'ragged_rank', 0))
        self.assertAllEqual(result, expected)

    @parameterized.parameters([dict(doc='x.shape=[3, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]', x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]], dtype=np.int32), y=[[10], [20], [30]], expected=ragged_factory_ops.constant_value([[11, 12, 13], [], [34, 35]])), dict(doc='x.shape=[3, (D1)]; y.shape=[]; bcast.shape=[3, (D1)]', x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]], dtype=np.int32), y=10, expected=ragged_factory_ops.constant_value([[11, 12, 13], [], [14, 15]])), dict(doc='x.shape=[1, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]', x=ragged_factory_ops.constant_value([[1, 2, 3]], dtype=np.int32), y=[[10], [20], [30]], expected=ragged_factory_ops.constant_value([[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=np.int32)), dict(doc='x.shape=[2, (D1), 1]; y.shape=[1, (D2)]; bcast.shape=[2, (D1), (D2)]', x=ragged_factory_ops.constant_value([[[1], [2], [3]], [[4]]], ragged_rank=1), y=ragged_factory_ops.constant_value([[10, 20, 30]]), expected=ragged_factory_ops.constant_value([[[11, 21, 31], [12, 22, 32], [13, 23, 33]], [[14, 24, 34]]])), dict(doc='x.shape=[2, (D1), 1]; y.shape=[1, 1, 4]; bcast.shape=[2, (D1), 4]', x=ragged_factory_ops.constant_value([[[10], [20]], [[30]]], ragged_rank=1), y=[[[1, 2, 3, 4]]], expected=ragged_factory_ops.constant_value([[[11, 12, 13, 14], [21, 22, 23, 24]], [[31, 32, 33, 34]]], ragged_rank=1)), dict(doc='x.shape=[2, (D1), 2, 1]; y.shape=[2, (D2)]; bcast.shape=[2, (D1), (2), (D2)', x=ragged_factory_ops.constant_value([[[[1], [2]], [[3], [4]]], [[[5], [6]]]], ragged_rank=1), y=ragged_factory_ops.constant_value([[10, 20], [30]]), expected=ragged_factory_ops.constant_value([[[[11, 21], [32]], [[13, 23], [34]]], [[[15, 25], [36]]]]))])
    def testRaggedAddWithBroadcasting(self, x, y, expected, doc):
        if False:
            for i in range(10):
                print('nop')
        expected_rrank = getattr(expected, 'ragged_rank', 0)
        x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, dtype=dtypes.int32)
        y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, dtype=dtypes.int32)
        result = x + y
        result_rrank = getattr(result, 'ragged_rank', 0)
        self.assertEqual(expected_rrank, result_rrank)
        if hasattr(expected, 'tolist'):
            expected = expected.tolist()
        self.assertAllEqual(result, expected)
if __name__ == '__main__':
    googletest.main()