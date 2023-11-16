"""Tests for ragged_one_hot."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest

@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class RaggedOneHotTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters([dict(indices=[[0, 2, -1], [3]], depth=4, expected=[[[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], [[0, 0, 0, 1]]]), dict(indices=[[0, 2, -1], [3]], depth=4, axis=-1, expected=[[[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], [[0, 0, 0, 1]]]), dict(indices=[[0, 2, -1], [3]], depth=4, axis=2, expected=[[[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], [[0, 0, 0, 1]]]), dict(indices=[[0, 2, -1], [3]], depth=4, on_value=8, off_value=4, expected=[[[8, 4, 4, 4], [4, 4, 8, 4], [4, 4, 4, 4]], [[4, 4, 4, 8]]]), dict(indices=[[0, 2, -1], [3]], depth=0, expected=[[[], [], []], [[]]]), dict(indices=[[[0, 2, -1], [3]], [[2, 8]]], depth=4, expected=[[[[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], [[0, 0, 0, 1]]], [[[0, 0, 1, 0], [0, 0, 0, 0]]]]), dict(indices=[[[0, 2], [-1, 3]], [[2, 8]]], ragged_rank=1, depth=4, expected=[[[[1, 0, 0, 0], [0, 0, 1, 0]], [[0, 0, 0, 0], [0, 0, 0, 1]]], [[[0, 0, 1, 0], [0, 0, 0, 0]]]]), dict(indices=[[[0, 2], [-1, 3]], [[2, 8]]], ragged_rank=1, axis=2, depth=4, expected=[[[[1, 0], [0, 0], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 1]]], [[[0, 0], [0, 0], [1, 0], [0, 0]]]])])
    def testRaggedOneHot(self, indices, depth, on_value=None, off_value=None, axis=None, dtype=None, expected=None, ragged_rank=None):
        if False:
            return 10
        ragged_indices = ragged_factory_ops.constant(indices, ragged_rank=ragged_rank)
        result = ragged_array_ops.ragged_one_hot(ragged_indices, depth, on_value=on_value, off_value=off_value, axis=axis, dtype=dtype)
        self.assertAllEqual(result, expected)
        self.assertEqual(result.ragged_rank, ragged_indices.ragged_rank)

    @parameterized.parameters([dict(indices=[[1]], depth=4, axis=0, message='axis \\(0\\) must be greater than indices.ragged_rank'), dict(indices=[[1]], depth=4, axis=1, message='axis \\(1\\) must be greater than indices.ragged_rank'), dict(indices=[[1]], depth=4, axis=-2, message='(?i)axis must be >= -1|Expected axis.* to be -1 or between.*', exception=(ValueError, errors.InvalidArgumentError, errors.UnknownError))])
    def testErrors(self, indices, depth, on_value=None, off_value=None, axis=None, dtype=None, exception=ValueError, message=None, ragged_rank=None):
        if False:
            while True:
                i = 10
        ragged_indices = ragged_factory_ops.constant(indices, ragged_rank=ragged_rank)
        with self.assertRaisesRegex(exception, message):
            array_ops.one_hot(ragged_indices, depth, on_value=on_value, off_value=off_value, axis=axis, dtype=dtype)

    @parameterized.parameters([dict(indices_shape=[5, 7], depth=6, axis=-1), dict(indices_shape=[5, 7], depth=6, axis=2), dict(indices_shape=[5, 2, 7], depth=3, axis=-1), dict(indices_shape=[5, 2, 7], depth=3, axis=3), dict(indices_shape=[5, 2, 7], depth=3, axis=2), dict(indices_shape=[5, 2, 7, 4], depth=3, axis=-1), dict(indices_shape=[5, 2, 7, 4], depth=3, axis=4), dict(indices_shape=[5, 2, 7, 4], depth=3, axis=3), dict(indices_shape=[5, 2, 7, 4], depth=3, axis=2), dict(indices_shape=[5, 2, 7], depth=3, on_value=True, off_value=False), dict(indices_shape=[5, 2, 7], depth=3, dtype=dtypes.float32)])
    def testRaggedOneHotMatchesArrayOpsOneHot(self, indices_shape, depth, on_value=None, off_value=None, axis=None, dtype=None):
        if False:
            return 10
        'Tests that tf.one_hot gives the same result for ragged & uniform tensors.\n\n    Runs tf.one_hot with a uniform tensor, and compares the output with the\n    results of calling tf.one_hot with ragged version of that tensor with\n    varying ragged ranks.\n\n    Args:\n      indices_shape: Shape for `indices` arg to `tf.one_hot`\n      depth: `depth` arg to `tf.one_hot`\n      on_value: `on_value` arg to `tf.one_hot`\n      off_value: `off_value` arg to `tf.one_hot`\n      axis: `axis` arg to `tf.one_hot`\n      dtype: `dtype` arg to `tf.one_hot`\n    '
        indices_shape = tensor_shape.as_shape(indices_shape)
        indices = np.random.randint(depth + 1, size=indices_shape)
        expected = array_ops.one_hot(indices, depth, on_value=on_value, off_value=off_value, axis=axis, dtype=dtype)
        for ragged_rank in range(1, len(indices_shape)):
            if axis is not None and 0 <= axis <= ragged_rank:
                continue
            ragged_indices = ragged_tensor.RaggedTensor.from_tensor(indices, ragged_rank=ragged_rank)
            result = ragged_array_ops.ragged_one_hot(ragged_indices, depth, on_value=on_value, off_value=off_value, axis=axis, dtype=dtype)
            self.assertAllEqual(result.to_tensor(), expected)
if __name__ == '__main__':
    googletest.main()