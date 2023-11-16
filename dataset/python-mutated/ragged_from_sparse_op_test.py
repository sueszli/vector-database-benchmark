"""Tests for RaggedTensor.from_sparse."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorToSparseOpTest(test_util.TensorFlowTestCase):

    def testDocStringExample(self):
        if False:
            while True:
                i = 10
        st = sparse_tensor.SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [3, 0]], values=[1, 2, 3, 4, 5], dense_shape=[4, 3])
        rt = RaggedTensor.from_sparse(st)
        self.assertAllEqual(rt, [[1, 2, 3], [4], [], [5]])

    def testEmpty(self):
        if False:
            i = 10
            return i + 15
        st = sparse_tensor.SparseTensor(indices=array_ops.zeros([0, 2], dtype=dtypes.int64), values=[], dense_shape=[4, 3])
        rt = RaggedTensor.from_sparse(st)
        self.assertAllEqual(rt, [[], [], [], []])

    def testBadSparseTensorRank(self):
        if False:
            while True:
                i = 10
        st1 = sparse_tensor.SparseTensor(indices=[[0]], values=[0], dense_shape=[3])
        self.assertRaisesRegex(ValueError, 'rank\\(st_input\\) must be 2', RaggedTensor.from_sparse, st1)
        st2 = sparse_tensor.SparseTensor(indices=[[0, 0, 0]], values=[0], dense_shape=[3, 3, 3])
        self.assertRaisesRegex(ValueError, 'rank\\(st_input\\) must be 2', RaggedTensor.from_sparse, st2)
        if not context.executing_eagerly():
            st3 = sparse_tensor.SparseTensor(indices=array_ops.placeholder(dtypes.int64), values=[0], dense_shape=array_ops.placeholder(dtypes.int64))
            self.assertRaisesRegex(ValueError, 'rank\\(st_input\\) must be 2', RaggedTensor.from_sparse, st3)

    def testGoodPartialSparseTensorRank(self):
        if False:
            print('Hello World!')
        if not context.executing_eagerly():
            st1 = sparse_tensor.SparseTensor(indices=[[0, 0]], values=[0], dense_shape=array_ops.placeholder(dtypes.int64))
            st2 = sparse_tensor.SparseTensor(indices=array_ops.placeholder(dtypes.int64), values=[0], dense_shape=[4, 3])
            RaggedTensor.from_sparse(st1)
            RaggedTensor.from_sparse(st2)

    def testNonRaggedSparseTensor(self):
        if False:
            return 10
        st1 = sparse_tensor.SparseTensor(indices=[[0, 1], [0, 2], [2, 0]], values=[1, 2, 3], dense_shape=[3, 3])
        with self.assertRaisesRegex(errors.InvalidArgumentError, '.*SparseTensor is not right-ragged'):
            self.evaluate(RaggedTensor.from_sparse(st1))
        st2 = sparse_tensor.SparseTensor(indices=[[0, 0], [0, 1], [2, 1]], values=[1, 2, 3], dense_shape=[3, 3])
        with self.assertRaisesRegex(errors.InvalidArgumentError, '.*SparseTensor is not right-ragged'):
            self.evaluate(RaggedTensor.from_sparse(st2))
        st3 = sparse_tensor.SparseTensor(indices=[[0, 1], [0, 1], [0, 3]], values=[1, 2, 3], dense_shape=[3, 3])
        with self.assertRaisesRegex(errors.InvalidArgumentError, '.*SparseTensor is not right-ragged'):
            self.evaluate(RaggedTensor.from_sparse(st3))
if __name__ == '__main__':
    googletest.main()