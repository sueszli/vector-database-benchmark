"""Tests for SparseConcat."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test

class SparseConcatTest(test.TestCase):

    def _SparseTensor_UnknownShape(self, ind_shape=None, val_shape=None, shape_shape=None):
        if False:
            while True:
                i = 10
        return sparse_tensor.SparseTensor(array_ops.placeholder(dtypes.int64, shape=ind_shape), array_ops.placeholder(dtypes.float32, shape=val_shape), array_ops.placeholder(dtypes.int64, shape=shape_shape))

    def _SparseTensorValue_3x3(self):
        if False:
            return 10
        ind = np.array([[0, 2], [1, 0], [2, 0], [2, 2]])
        val = np.array([1, 2, 3, 4])
        shape = np.array([3, 3])
        return sparse_tensor.SparseTensorValue(np.array(ind, np.int64), np.array(val, np.float32), np.array(shape, np.int64))

    def _SparseTensor_3x3(self):
        if False:
            while True:
                i = 10
        return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_3x3())

    def _SparseTensorValue_3x5(self):
        if False:
            print('Hello World!')
        ind = np.array([[1, 1], [2, 0], [2, 3], [2, 4]])
        val = np.array([1, 2, 1, 0])
        shape = np.array([3, 5])
        return sparse_tensor.SparseTensorValue(np.array(ind, np.int64), np.array(val, np.float32), np.array(shape, np.int64))

    def _SparseTensor_3x5(self):
        if False:
            return 10
        return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_3x5())

    def _SparseTensor_3x2(self):
        if False:
            print('Hello World!')
        ind = np.array([[1, 0], [2, 0]])
        val = np.array([1, 2])
        shape = np.array([3, 2])
        return sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64), constant_op.constant(val, dtypes.float32), constant_op.constant(shape, dtypes.int64))

    def _SparseTensor_2x3(self):
        if False:
            print('Hello World!')
        ind = np.array([[0, 1], [1, 0], [1, 2]])
        val = np.array([1, 1, 2])
        shape = np.array([2, 3])
        return sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64), constant_op.constant(val, dtypes.float32), constant_op.constant(shape, dtypes.int64))

    def _SparseTensor_2x3x4(self):
        if False:
            for i in range(10):
                print('nop')
        ind = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 2], [1, 0, 3], [1, 1, 1], [1, 1, 3], [1, 2, 2]])
        val = np.array([1, 10, 12, 103, 111, 113, 122])
        shape = np.array([2, 3, 4])
        return sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64), constant_op.constant(val, dtypes.float32), constant_op.constant(shape, dtypes.int64))

    def _SparseTensor_NoNonZeros(self, dense_shape):
        if False:
            return 10
        ind = np.empty(shape=(0, len(dense_shape)))
        val = np.array([])
        shape = np.array(dense_shape)
        return sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64), constant_op.constant(val, dtypes.float32), constant_op.constant(shape, dtypes.int64))

    def _SparseTensor_String3x3(self):
        if False:
            return 10
        ind = np.array([[0, 2], [1, 0], [2, 0], [2, 2]])
        val = np.array(['a', 'b', 'c', 'd'])
        shape = np.array([3, 3])
        return sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64), constant_op.constant(val, dtypes.string), constant_op.constant(shape, dtypes.int64))

    def _SparseTensor_String3x5(self):
        if False:
            for i in range(10):
                print('nop')
        ind = np.array([[1, 1], [2, 0], [2, 3], [2, 4]])
        val = np.array(['e', 'f', 'g', 'h'])
        shape = np.array([3, 5])
        return sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64), constant_op.constant(val, dtypes.string), constant_op.constant(shape, dtypes.int64))

    def testConcat1(self):
        if False:
            while True:
                i = 10
        with self.session() as sess:
            for sp_a in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
                for concat_dim in (-2000, 1, 2000):
                    sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a])
                    self.assertEqual(sp_concat.indices.get_shape(), [4, 2])
                    self.assertEqual(sp_concat.values.get_shape(), [4])
                    self.assertEqual(sp_concat.dense_shape.get_shape(), [2])
                    concat_out = self.evaluate(sp_concat)
                    self.assertAllEqual(concat_out.indices, [[0, 2], [1, 0], [2, 0], [2, 2]])
                    self.assertAllEqual(concat_out.values, [1, 2, 3, 4])
                    self.assertAllEqual(concat_out.dense_shape, [3, 3])

    def testConcat2(self):
        if False:
            return 10
        with self.session() as sess:
            for sp_a in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
                for sp_b in (self._SparseTensorValue_3x5(), self._SparseTensor_3x5()):
                    for concat_dim in (-1, 1):
                        sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b])
                        self.assertEqual(sp_concat.indices.get_shape(), [8, 2])
                        self.assertEqual(sp_concat.values.get_shape(), [8])
                        self.assertEqual(sp_concat.dense_shape.get_shape(), [2])
                        concat_out = self.evaluate(sp_concat)
                        self.assertAllEqual(concat_out.indices, [[0, 2], [1, 0], [1, 4], [2, 0], [2, 2], [2, 3], [2, 6], [2, 7]])
                        self.assertAllEqual(concat_out.values, [1, 2, 1, 3, 4, 2, 1, 0])
                        self.assertAllEqual(concat_out.dense_shape, [3, 8])

    def testConcatDim0(self):
        if False:
            i = 10
            return i + 15
        with self.session() as sess:
            sp_a = self._SparseTensor_3x3()
            sp_d = self._SparseTensor_2x3()
            for concat_dim in (-2, 0):
                sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_d])
                self.assertEqual(sp_concat.indices.get_shape(), [7, 2])
                self.assertEqual(sp_concat.values.get_shape(), [7])
                self.assertEqual(sp_concat.dense_shape.get_shape(), [2])
                concat_out = self.evaluate(sp_concat)
                self.assertAllEqual(concat_out.indices, [[0, 2], [1, 0], [2, 0], [2, 2], [3, 1], [4, 0], [4, 2]])
                self.assertAllEqual(concat_out.values, np.array([1, 2, 3, 4, 1, 1, 2]))
                self.assertAllEqual(concat_out.dense_shape, np.array([5, 3]))

    def testConcat3(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as sess:
            sp_a = self._SparseTensor_3x3()
            sp_b = self._SparseTensor_3x5()
            sp_c = self._SparseTensor_3x2()
            for concat_dim in (-1, 1):
                sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b, sp_c])
                self.assertEqual(sp_concat.indices.get_shape(), [10, 2])
                self.assertEqual(sp_concat.values.get_shape(), [10])
                self.assertEqual(sp_concat.dense_shape.get_shape(), [2])
                concat_out = self.evaluate(sp_concat)
                self.assertAllEqual(concat_out.indices, [[0, 2], [1, 0], [1, 4], [1, 8], [2, 0], [2, 2], [2, 3], [2, 6], [2, 7], [2, 8]])
                self.assertAllEqual(concat_out.values, [1, 2, 1, 1, 3, 4, 2, 1, 0, 2])
                self.assertAllEqual(concat_out.dense_shape, [3, 10])

    def testConcatNoNonZeros(self):
        if False:
            print('Hello World!')
        sp_a = self._SparseTensor_NoNonZeros((2, 3, 4))
        sp_b = self._SparseTensor_NoNonZeros((2, 7, 4))
        sp_c = self._SparseTensor_NoNonZeros((2, 5, 4))
        with self.session() as sess:
            concat_dim = 1
            sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b, sp_c])
            self.assertEqual(sp_concat.indices.get_shape(), [0, 3])
            self.assertEqual(sp_concat.values.get_shape(), [0])
            self.assertEqual(sp_concat.dense_shape.get_shape(), [3])
            concat_out = self.evaluate(sp_concat)
            self.assertEqual(concat_out.indices.shape, (0, 3))
            self.assertEqual(concat_out.values.shape, (0,))
            self.assertAllEqual(concat_out.dense_shape, [2, 15, 4])

    def testConcatSomeNoNonZeros(self):
        if False:
            i = 10
            return i + 15
        sp_a = self._SparseTensor_NoNonZeros((2, 7, 4))
        sp_b = self._SparseTensor_2x3x4()
        sp_c = self._SparseTensor_NoNonZeros((2, 5, 4))
        output_nnz = sp_b.indices.get_shape()[0]
        with self.session() as sess:
            concat_dim = 1
            sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b, sp_c])
            self.assertEqual(sp_concat.indices.get_shape(), [output_nnz, 3])
            self.assertEqual(sp_concat.values.get_shape(), [output_nnz])
            self.assertEqual(sp_concat.dense_shape.get_shape(), [3])
            concat_out = self.evaluate(sp_concat)
            self.assertAllEqual(concat_out.indices, sp_b.indices + [0, sp_a.dense_shape[1], 0])
            self.assertAllEqual(concat_out.values, sp_b.values)
            self.assertAllEqual(concat_out.dense_shape, [2, 15, 4])

    def testConcatNonNumeric(self):
        if False:
            return 10
        with self.session(use_gpu=False) as sess:
            sp_a = self._SparseTensor_String3x3()
            sp_b = self._SparseTensor_String3x5()
            for concat_dim in (-1, 1):
                sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b])
                self.assertEqual(sp_concat.indices.get_shape(), [8, 2])
                self.assertEqual(sp_concat.values.get_shape(), [8])
                self.assertEqual(sp_concat.dense_shape.get_shape(), [2])
                concat_out = self.evaluate(sp_concat)
                self.assertAllEqual(concat_out.indices, [[0, 2], [1, 0], [1, 4], [2, 0], [2, 2], [2, 3], [2, 6], [2, 7]])
                self.assertAllEqual(concat_out.values, [b'a', b'b', b'e', b'c', b'd', b'f', b'g', b'h'])
                self.assertAllEqual(concat_out.dense_shape, [3, 8])

    @test_util.run_deprecated_v1
    def testMismatchedRank(self):
        if False:
            return 10
        with self.session():
            sp_a = self._SparseTensor_3x3()
            sp_e = self._SparseTensor_2x3x4()
            for concat_dim in (-1, 1):
                with self.assertRaises(ValueError):
                    sparse_ops.sparse_concat(concat_dim, [sp_a, sp_e])

    @test_util.run_deprecated_v1
    def testMismatchedRankExpandNonconcatDim(self):
        if False:
            return 10
        with self.session():
            sp_a = self._SparseTensor_3x3()
            sp_e = self._SparseTensor_2x3x4()
            for concat_dim in (-1, 1):
                with self.assertRaises(ValueError):
                    sparse_ops.sparse_concat(concat_dim, [sp_a, sp_e], expand_nonconcat_dim=True)

    @test_util.run_deprecated_v1
    def testMismatchedShapes(self):
        if False:
            i = 10
            return i + 15
        with self.session() as sess:
            sp_a = self._SparseTensor_3x3()
            sp_b = self._SparseTensor_3x5()
            sp_c = self._SparseTensor_3x2()
            sp_d = self._SparseTensor_2x3()
            for concat_dim in (-1, 1):
                sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b, sp_c, sp_d])
                with self.assertRaisesOpError('Input shapes must match'):
                    self.evaluate(sp_concat)

    def testMismatchedShapesExpandNonconcatDim(self):
        if False:
            print('Hello World!')
        with self.session() as sess:
            sp_a = self._SparseTensor_3x3()
            sp_b = self._SparseTensor_3x5()
            sp_c = self._SparseTensor_3x2()
            sp_d = self._SparseTensor_2x3()
            for concat_dim0 in (-2, 0):
                for concat_dim1 in (-1, 1):
                    sp_concat_dim0 = sparse_ops.sparse_concat(concat_dim0, [sp_a, sp_b, sp_c, sp_d], expand_nonconcat_dim=True)
                    sp_concat_dim1 = sparse_ops.sparse_concat(concat_dim1, [sp_a, sp_b, sp_c, sp_d], expand_nonconcat_dim=True)
                    sp_concat_dim0_out = self.evaluate(sp_concat_dim0)
                    sp_concat_dim1_out = self.evaluate(sp_concat_dim1)
                    self.assertAllEqual(sp_concat_dim0_out.indices, [[0, 2], [1, 0], [2, 0], [2, 2], [4, 1], [5, 0], [5, 3], [5, 4], [7, 0], [8, 0], [9, 1], [10, 0], [10, 2]])
                    self.assertAllEqual(sp_concat_dim0_out.values, [1, 2, 3, 4, 1, 2, 1, 0, 1, 2, 1, 1, 2])
                    self.assertAllEqual(sp_concat_dim0_out.dense_shape, [11, 5])
                    self.assertAllEqual(sp_concat_dim1_out.indices, [[0, 2], [0, 11], [1, 0], [1, 4], [1, 8], [1, 10], [1, 12], [2, 0], [2, 2], [2, 3], [2, 6], [2, 7], [2, 8]])
                    self.assertAllEqual(sp_concat_dim1_out.values, [1, 1, 2, 1, 1, 1, 2, 3, 4, 2, 1, 0, 2])
                    self.assertAllEqual(sp_concat_dim1_out.dense_shape, [3, 13])

    @test_util.run_deprecated_v1
    def testShapeInferenceUnknownShapes(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            sp_inputs = [self._SparseTensor_UnknownShape(), self._SparseTensor_UnknownShape(val_shape=[3]), self._SparseTensor_UnknownShape(ind_shape=[1, 3]), self._SparseTensor_UnknownShape(shape_shape=[3])]
            for concat_dim in (-2, 0):
                sp_concat = sparse_ops.sparse_concat(concat_dim, sp_inputs)
                self.assertEqual(sp_concat.indices.get_shape().as_list(), [None, 3])
                self.assertEqual(sp_concat.values.get_shape().as_list(), [None])
                self.assertEqual(sp_concat.dense_shape.get_shape(), [3])

    def testConcatShape(self):
        if False:
            while True:
                i = 10
        x = sparse_tensor.SparseTensor(indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[2, 2])
        y = sparse_tensor.SparseTensor(indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[2, 2])
        z = sparse_ops.sparse_concat(-1, [x, y])
        self.assertEqual(z.get_shape().as_list(), [2, 4])
if __name__ == '__main__':
    test.main()