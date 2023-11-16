"""Tests for tensorflow.kernels.sparse_op."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test

@test_util.with_eager_op_as_function
class SparseToDenseTest(test.TestCase, parameterized.TestCase):

    def testInt(self):
        if False:
            for i in range(10):
                print('nop')
        tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], 1, 0)
        np_ans = np.array([0, 1, 0, 1, 0]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

    @parameterized.parameters(dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64)
    def testFloatTypes(self, dtype):
        if False:
            return 10
        tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], array_ops.constant(1.0, dtype=dtype), 0.0)
        np_ans = np.array([0, 1, 0, 1, 0]).astype(dtype.as_numpy_dtype)
        self.assertAllClose(np_ans, tf_ans)

    def testComplex(self):
        if False:
            return 10
        for dtype in [dtypes.complex64, dtypes.complex128]:
            tf_val = math_ops.cast(constant_op.constant([1.0 + 1j, 2.0 - 2j]), dtypes.complex128)
            tf_ans = sparse_ops.sparse_tensor_to_dense(sparse_ops.from_dense(tf_val))
            self.assertAllClose(tf_val, tf_ans)

    def testEmptyNonZeros(self):
        if False:
            i = 10
            return i + 15
        indices = array_ops.constant([], dtype=dtypes.int32)
        values = array_ops.constant([], dtype=dtypes.float32)
        tf_ans = sparse_ops.sparse_to_dense(indices, [5], values, 0.0)
        np_ans = np.array([0, 0, 0, 0, 0]).astype(np.float32)
        self.assertAllClose(np_ans, tf_ans)

    def testString(self):
        if False:
            return 10
        tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], 'a', 'b')
        np_ans = np.array(['b', 'a', 'b', 'a', 'b']).astype(np.string_)
        self.assertAllEqual(np_ans, tf_ans)

    def testSetValue(self):
        if False:
            while True:
                i = 10
        tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], [1, 2], -1)
        np_ans = np.array([-1, 1, -1, 2, -1]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

    def testSetSingleValue(self):
        if False:
            return 10
        tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], 1, -1)
        np_ans = np.array([-1, 1, -1, 1, -1]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

    def test2d(self):
        if False:
            print('Hello World!')
        tf_ans = sparse_ops.sparse_to_dense([[1, 3], [2, 0]], [3, 4], 1, -1)
        np_ans = np.array([[-1, -1, -1, -1], [-1, -1, -1, 1], [1, -1, -1, -1]]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

    def testZeroDefault(self):
        if False:
            while True:
                i = 10
        x = sparse_ops.sparse_to_dense(2, [4], 7)
        self.assertAllEqual(x, [0, 0, 7, 0])

    def test3d(self):
        if False:
            for i in range(10):
                print('nop')
        tf_ans = sparse_ops.sparse_to_dense([[1, 3, 0], [2, 0, 1]], [3, 4, 2], 1, -1)
        np_ans = np.ones((3, 4, 2), dtype=np.int32) * -1
        np_ans[1, 3, 0] = 1
        np_ans[2, 0, 1] = 1
        self.assertAllClose(np_ans, tf_ans)

    def testBadShape(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'must be rank 1'):
            sparse_ops.sparse_to_dense([1, 3], [[5], [3]], 1, -1)

    def testBadValue(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'sparse_values has incorrect shape \\[2,1\\], should be \\[\\] or \\[2\\]'):
            self.evaluate(sparse_ops.sparse_to_dense([1, 3], [5], [[5], [3]], -1))

    def testBadNumValues(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'sparse_values has incorrect shape \\[3\\], should be \\[\\] or \\[2\\]'):
            self.evaluate(sparse_ops.sparse_to_dense([1, 3], [5], [1, 2, 3], -1))

    def testBadDefault(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'default_value should be a scalar'):
            self.evaluate(sparse_ops.sparse_to_dense([1, 3], [5], [1, 2], [0]))

    @test_util.disable_xla('XLA does not check validity for SparseToDense')
    def testOutOfBoundsIndicesWithWithoutValidation(self):
        if False:
            return 10
        error_msg = 'out of bounds' if test_util.is_gpu_available() else 'indices\\[1\\] = \\[10\\] is out of bounds: need 0 <= index < \\[5\\]'
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), error_msg):
            self.evaluate(sparse_ops.sparse_to_dense([[1], [10]], [5], [1.0, 1.0], 0.0))
        if not test_util.is_gpu_available():
            with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'out of bounds'):
                self.evaluate(sparse_ops.sparse_to_dense([[1], [10]], [5], [-1.0, 1.0], 0.0, validate_indices=False))

    @test_util.disable_xla('XLA does not check validity for SparseToDense')
    def testRepeatingIndicesWithWithoutValidation(self):
        if False:
            while True:
                i = 10
        error_msg = 'indices\\[1\\] is repeated' if test_util.is_gpu_available() else 'indices\\[1\\] = \\[1\\] is repeated'
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), error_msg):
            self.evaluate(sparse_ops.sparse_to_dense([[1], [1]], [5], [-1.0, 1.0], 0.0))
        self.evaluate(sparse_ops.sparse_to_dense([[1], [1]], [5], [-1.0, 1.0], 0.0, validate_indices=False))

    @test_util.disable_xla('XLA does not check validity for SparseToDense')
    def testUnsortedIndicesWithWithoutValidation(self):
        if False:
            print('Hello World!')
        error_msg = 'indices\\[1\\] is out of order' if test_util.is_gpu_available() else 'indices\\[1\\] = \\[1\\] is out of order'
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), error_msg):
            self.evaluate(sparse_ops.sparse_to_dense([[2], [1]], [5], [-1.0, 1.0], 0.0))
        self.evaluate(sparse_ops.sparse_to_dense([[2], [1]], [5], [-1.0, 1.0], 0.0, validate_indices=False))

    def testShapeInferenceKnownShape(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            indices = array_ops.placeholder(dtypes.int64)
            shape = [4, 5, 6]
            output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
            self.assertEqual(output.get_shape(), [4, 5, 6])
            shape = array_ops.placeholder(dtypes.int64, shape=(3,))
            output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
            self.assertEqual(output.get_shape().as_list(), [None, None, None])

    def testShapeInferenceUnknownShape(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            indices = array_ops.placeholder(dtypes.int64)
            shape = array_ops.placeholder(dtypes.int64)
            output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
            self.assertIsNone(output.get_shape().ndims)
if __name__ == '__main__':
    test.main()