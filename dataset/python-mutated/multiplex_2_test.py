"""Tests for multiplex_2."""
import numpy as np
import tensorflow as tf
from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util

@test_util.with_eager_op_as_function
class MultiplexOpRank1Test(tf.test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_multiplex_int(self):
        if False:
            for i in range(10):
                print('nop')
        a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
        b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
        cond = tf.constant([True, False, True, False, True], dtype=bool)
        expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
        result = multiplex_2_op.multiplex(cond, a, b)
        self.assertAllEqual(result, expect)

    @test_util.run_in_graph_and_eager_modes
    def test_multiplex_float(self):
        if False:
            while True:
                i = 10
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        b = tf.constant([10.0, 20.0, 30.0, 40.0, 50.0])
        cond = tf.constant([True, False, True, False, True], dtype=bool)
        expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
        result = multiplex_2_op.multiplex(cond, a, b)
        self.assertAllEqual(result, expect)

    @test_util.run_in_graph_and_eager_modes
    def test_multiplex_bad_types(self):
        if False:
            while True:
                i = 10
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
        cond = tf.constant([True, False, True, False, True], dtype=bool)
        with self.assertRaisesRegex((errors_impl.InvalidArgumentError, TypeError), "(cannot compute Examples>MultiplexDense as input #2\\(zero-based\\) was expected to be a float tensor but is a int64 tensor \\[Op:Examples>MultiplexDense\\])|(Input 'b' of 'Examples>MultiplexDense' Op has type int64 that does not match type float32 of argument 'a'.)"):
            self.evaluate(multiplex_2_op.multiplex(cond, a, b))

    @test_util.run_in_graph_and_eager_modes
    def test_multiplex_bad_size(self):
        if False:
            print('Hello World!')
        a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
        b = tf.constant([10, 20], dtype=tf.int64)
        cond = tf.constant([True, False, True, False, True], dtype=bool)
        with self.assertRaisesRegex((errors_impl.InvalidArgumentError, ValueError), '(?s)(a and b must have the same shape. a shape: \\[5\\] b shape: \\[2\\].* \\[Op:Examples>MultiplexDense\\])|(Dimension 0 in both shapes must be equal, but are 5 and 2\\. Shapes are \\[5\\] and \\[2\\]\\.)'):
            self.evaluate(multiplex_2_op.multiplex(cond, a, b))

    @test_util.run_in_graph_and_eager_modes
    def test_multiplex_2d(self):
        if False:
            return 10
        a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64)
        b = tf.constant([[10, 20, 30], [40, 50, 60]], dtype=tf.int64)
        cond = tf.constant([[True, False, True], [False, True, False]], dtype=bool)
        expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
        result = multiplex_2_op.multiplex(cond, a, b)
        self.assertAllEqual(result, expect)

    @test_util.run_in_graph_and_eager_modes
    def test_multiplex_bad_shape(self):
        if False:
            i = 10
            return i + 15
        a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64)
        b = tf.constant([[10, 20], [30, 40], [50, 60]], dtype=tf.int64)
        cond = tf.constant([[True, False, True], [False, True, False]], dtype=bool)
        with self.assertRaisesRegex((errors_impl.InvalidArgumentError, ValueError), '(a and b must have the same shape. a shape: \\[2,3\\] b shape: \\[3,2\\])|(Dimension 0 in both shapes must be equal, but are 2 and 3\\. Shapes are \\[2,3\\] and \\[3,2\\])\\.'):
            self.evaluate(multiplex_2_op.multiplex(cond, a, b))
if __name__ == '__main__':
    tf.test.main()