"""Tests for multiplex_3."""
import numpy as np
import tensorflow as tf
from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op
from tensorflow.examples.custom_ops_doc.multiplex_3 import multiplex_3_op
from tensorflow.python.framework import test_util

@test_util.with_eager_op_as_function
class MultiplexOpRank1Test(tf.test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_sparse_kernel(self):
        if False:
            for i in range(10):
                print('nop')
        idx0 = tf.constant([], dtype=tf.int64, shape=[0, 1])
        val0 = tf.constant([], dtype=tf.int64)
        val5a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
        idx5b = tf.constant([[10], [20], [30], [40], [50]], dtype=tf.int64)
        val5b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
        cond0 = tf.constant([], dtype=bool)
        cond5 = tf.constant([True, False, True, False, True], dtype=bool)
        val3a = tf.constant([1, 2, 3], dtype=tf.int64)
        val3b = tf.constant([4, 5, 6], dtype=tf.int64)
        idx3c = tf.constant([[10], [20], [30]], dtype=tf.int64)
        idx3d = tf.constant([[30], [40], [50]], dtype=tf.int64)
        idx3e = tf.constant([[10], [30], [50]], dtype=tf.int64)
        cond3 = tf.constant([True, False, True], dtype=bool)
        shape = tf.constant([100], dtype=tf.int64)
        (result_index, result_values, result_shape) = multiplex_3_op.examples_multiplex_sparse(idx0, cond0, shape, idx0, val0, shape, idx0, val0, shape)
        self.assertAllEqual(idx0, result_index)
        self.assertAllEqual(val0, result_values)
        self.assertAllEqual(shape, result_shape)
        (result_index, result_values, result_shape) = multiplex_3_op.examples_multiplex_sparse(idx0, cond0, shape, idx0, val0, shape, idx5b, val5a, shape)
        self.assertAllEqual(idx5b, result_index)
        self.assertAllEqual(val5a, result_values)
        self.assertAllEqual(shape, result_shape)
        (result_index, result_values, result_shape) = multiplex_3_op.examples_multiplex_sparse(idx5b, cond5, shape, idx5b, val5a, shape, idx5b, val5b, shape)
        expect_values = tf.constant([1, 20, 3, 40, 5], dtype=tf.int64)
        self.assertAllEqual(idx5b, result_index)
        self.assertAllEqual(expect_values, result_values)
        self.assertAllEqual(shape, result_shape)
        (result_index, result_values, result_shape) = multiplex_3_op.examples_multiplex_sparse(idx3c, cond3, shape, idx3c, val3a, shape, idx3d, val3b, shape)
        expect_index = tf.constant([[10], [30], [40], [50]], dtype=tf.int64)
        expect_values = tf.constant([1, 3, 5, 6], dtype=tf.int64)
        self.assertAllEqual(expect_index, result_index)
        self.assertAllEqual(expect_values, result_values)
        self.assertAllEqual(shape, result_shape)
        (result_index, result_values, result_shape) = multiplex_3_op.examples_multiplex_sparse(idx3d, cond3, shape, idx3d, val3a, shape, idx3c, val3b, shape)
        expect_index = tf.constant([[10], [20], [30], [50]], dtype=tf.int64)
        expect_values = tf.constant([4, 5, 1, 3], dtype=tf.int64)
        self.assertAllEqual(expect_index, result_index)
        self.assertAllEqual(expect_values, result_values)
        self.assertAllEqual(shape, result_shape)
        (result_index, result_values, result_shape) = multiplex_3_op.examples_multiplex_sparse(idx3d, cond3, shape, idx3c, val3a, shape, idx3d, val3b, shape)
        expect_index = tf.constant([[30], [40]], dtype=tf.int64)
        expect_values = tf.constant([3, 5], dtype=tf.int64)
        self.assertAllEqual(expect_index, result_index)
        self.assertAllEqual(expect_values, result_values)
        self.assertAllEqual(shape, result_shape)
        (result_index, result_values, result_shape) = multiplex_3_op.examples_multiplex_sparse(idx3c, cond3, shape, idx3d, val3a, shape, idx3c, val3b, shape)
        expect_index = tf.constant([[20], [30]], dtype=tf.int64)
        expect_values = tf.constant([5, 1], dtype=tf.int64)
        self.assertAllEqual(expect_index, result_index)
        self.assertAllEqual(expect_values, result_values)
        self.assertAllEqual(shape, result_shape)
        (result_index, result_values, result_shape) = multiplex_3_op.examples_multiplex_sparse(idx3e, cond3, shape, idx3c, val3a, shape, idx3d, val3b, shape)
        expect_index = tf.constant([[10], [30], [40]], dtype=tf.int64)
        expect_values = tf.constant([1, 4, 5], dtype=tf.int64)
        self.assertAllEqual(expect_index, result_index)
        self.assertAllEqual(expect_values, result_values)
        self.assertAllEqual(shape, result_shape)

    @test_util.run_in_graph_and_eager_modes
    def test_sparse_op_only(self):
        if False:
            while True:
                i = 10
        cond = tf.SparseTensor(indices=[[1], [3], [6]], values=[True, False, True], dense_shape=[7])
        a = tf.SparseTensor(indices=[[1], [3], [5]], values=['a0', 'a1', 'a2'], dense_shape=[6])
        b = tf.SparseTensor(indices=[[0], [2], [3], [6]], values=['b0', 'b1', 'b2', 'b3'], dense_shape=[7])
        result = self.evaluate(multiplex_3_op.multiplex_sparse(cond, a, b))
        self.assertAllEqual([7], result.dense_shape)
        self.assertAllEqual([[0], [1], [2], [3]], result.indices)
        self.assertAllEqual([b'b0', b'a0', b'b1', b'b2'], result.values)

    @test_util.run_in_graph_and_eager_modes
    def test_sparse_op_same(self):
        if False:
            for i in range(10):
                print('nop')
        cond = tf.SparseTensor(indices=[[1], [3], [6]], values=[True, False, True], dense_shape=[7])
        a = tf.SparseTensor(indices=[[1], [3], [6]], values=['a0', 'a1', 'a2'], dense_shape=[6])
        b = tf.SparseTensor(indices=[[1], [3], [6]], values=['b0', 'b1', 'b2'], dense_shape=[7])
        result = self.evaluate(multiplex_2_op.multiplex(cond, a, b))
        self.assertAllEqual([7], result.dense_shape)
        self.assertAllEqual([[1], [3], [6]], result.indices)
        self.assertAllEqual([b'a0', b'b1', b'a2'], result.values)

    @test_util.run_in_graph_and_eager_modes
    def test_sparse_op_different(self):
        if False:
            return 10
        cond = tf.SparseTensor(indices=[[1], [3], [6]], values=[True, False, True], dense_shape=[7])
        a = tf.SparseTensor(indices=[[1], [3], [5]], values=['a0', 'a1', 'a2'], dense_shape=[6])
        b = tf.SparseTensor(indices=[[0], [2], [3], [6]], values=['b0', 'b1', 'b2', 'b3'], dense_shape=[7])
        result = self.evaluate(multiplex_2_op.multiplex(cond, a, b))
        self.assertAllEqual([7], result.dense_shape)
        self.assertAllEqual([[0], [1], [2], [3]], result.indices)
        self.assertAllEqual([b'b0', b'a0', b'b1', b'b2'], result.values)

    @test_util.run_in_graph_and_eager_modes
    def test_multiplex_int(self):
        if False:
            return 10
        a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
        b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
        cond = tf.constant([True, False, True, False, True], dtype=bool)
        expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
        result = multiplex_2_op.multiplex(cond, a, b)
        self.assertAllEqual(result, expect)
if __name__ == '__main__':
    tf.test.main()