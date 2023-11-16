"""Tests for tensorflow.kernels.sparse_op."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test

def _SparseToDense(sparse_indices, output_size, sparse_values, default_value, validate_indices=True):
    if False:
        i = 10
        return i + 15
    feed_sparse_indices = array_ops.placeholder(dtypes.int32)
    feed_dict = {feed_sparse_indices: sparse_indices}
    return sparse_ops.sparse_to_dense(feed_sparse_indices, output_size, sparse_values, default_value=default_value, validate_indices=validate_indices).eval(feed_dict=feed_dict)

class SparseToDenseTest(xla_test.XLATestCase):

    def testInt(self):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            tf_ans = _SparseToDense([1, 3], [5], 1, 0)
        np_ans = np.array([0, 1, 0, 1, 0]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

    def testFloat(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():
            tf_ans = _SparseToDense([1, 3], [5], 1.0, 0.0)
        np_ans = np.array([0, 1, 0, 1, 0]).astype(np.float32)
        self.assertAllClose(np_ans, tf_ans)

    def testSetValue(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():
            tf_ans = _SparseToDense([1, 3], [5], [1, 2], -1)
        np_ans = np.array([-1, 1, -1, 2, -1]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

    def testSetSingleValue(self):
        if False:
            return 10
        with self.session(), self.test_scope():
            tf_ans = _SparseToDense([1, 3], [5], 1, -1)
        np_ans = np.array([-1, 1, -1, 1, -1]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

    def test2d(self):
        if False:
            print('Hello World!')
        with self.session(), self.test_scope():
            tf_ans = _SparseToDense([[1, 3], [2, 0]], [3, 4], 1, -1)
        np_ans = np.array([[-1, -1, -1, -1], [-1, -1, -1, 1], [1, -1, -1, -1]]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

    def testZeroDefault(self):
        if False:
            while True:
                i = 10
        with self.session():
            x = sparse_ops.sparse_to_dense(2, [4], 7).eval()
            self.assertAllEqual(x, [0, 0, 7, 0])

    def test3d(self):
        if False:
            print('Hello World!')
        with self.session(), self.test_scope():
            tf_ans = _SparseToDense([[1, 3, 0], [2, 0, 1]], [3, 4, 2], 1, -1)
        np_ans = np.ones((3, 4, 2), dtype=np.int32) * -1
        np_ans[1, 3, 0] = 1
        np_ans[2, 0, 1] = 1
        self.assertAllClose(np_ans, tf_ans)

    def testDegenerateIndexMatrix(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():
            tf_ans = _SparseToDense([[2], [3], [4], [5], [6], [7], [8], [9]], [10], [1, 2, 3, 4, 5, 6, 7, 8], -1)
        self.assertAllClose([-1, -1, 1, 2, 3, 4, 5, 6, 7, 8], tf_ans)

    def testBadShape(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            with self.assertRaisesWithPredicateMatch(ValueError, 'must be rank 1'):
                _SparseToDense([1, 3], [[5], [3]], 1, -1)

    @test_util.disable_mlir_bridge('Error handling')
    def testBadValue(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            with self.assertRaisesOpError('sparse_values has incorrect shape \\[2,1\\], should be \\[\\] or \\[2\\]'):
                _SparseToDense([1, 3], [5], [[5], [3]], -1)

    @test_util.disable_mlir_bridge('Error handling')
    def testBadNumValues(self):
        if False:
            return 10
        with self.session(), self.test_scope():
            with self.assertRaisesOpError('sparse_values has incorrect shape \\[3\\], should be \\[\\] or \\[2\\]'):
                _SparseToDense([1, 3], [5], [1, 2, 3], -1)

    @test_util.disable_mlir_bridge('Error handling')
    def testBadDefault(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():
            with self.assertRaisesOpError('default_value should be a scalar'):
                _SparseToDense([1, 3], [5], [1, 2], [0])
if __name__ == '__main__':
    test.main()