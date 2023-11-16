"""Tests for multiplex_4."""
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.custom_ops_doc.multiplex_4 import model_using_multiplex
from tensorflow.examples.custom_ops_doc.multiplex_4 import multiplex_4_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util

@test_util.with_eager_op_as_function
class MultiplexOpTest(tf.test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_multiplex_int(self):
        if False:
            return 10
        a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
        b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
        cond = tf.constant([True, False, True, False, True], dtype=bool)
        expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
        result = multiplex_4_op.multiplex(cond, a, b)
        self.assertAllEqual(result, expect)

    @test_util.run_in_graph_and_eager_modes
    def test_multiplex_select(self):
        if False:
            while True:
                i = 10
        a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
        a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
        a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
        a = [a1, a2, a3]
        b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
        cond1 = tf.constant([False, False, True, False, False], dtype=bool)
        cond2 = tf.constant([False, False, False, False, True], dtype=bool)
        cond3 = tf.constant([True, False, True, False, True], dtype=bool)
        cond = [cond1, cond2, cond3]
        expect = np.select([self.evaluate(i) for i in cond], [self.evaluate(i) for i in a], self.evaluate(b))
        result = multiplex_4_op.multiplex(cond, a, b)
        self.assertAllEqual(result, expect)

    def test_multiplex_saved_model(self):
        if False:
            while True:
                i = 10
        path = os.path.join(self.create_tempdir(), 'model')
        model_using_multiplex.save(multiplex_4_op.multiplex, path)
        result = model_using_multiplex.load_and_use(path)
        self.assertAllEqual(result, tf.constant([1, 20, 3, 40, 5], dtype=tf.int64))

    @tf.function
    def _both(self):
        if False:
            for i in range(10):
                print('nop')
        a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
        a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
        a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
        a_123 = [a1, a2, a3]
        b_123 = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
        cond1 = tf.constant([False, False, True, False, False], dtype=bool)
        cond2 = tf.constant([False, False, False, False, True], dtype=bool)
        cond3 = tf.constant([True, False, True, False, True], dtype=bool)
        cond_123 = [cond1, cond2, cond3]
        mux_123 = multiplex_4_op.multiplex(cond_123, a_123, b_123)
        b4 = tf.constant([201, 202, 203, 204, 205], dtype=tf.int64)
        cond4 = tf.constant([True, True, True, False, False], dtype=bool)
        result = multiplex_4_op.multiplex(cond4, mux_123, b4)
        return result

    def test_both_single_and_list(self):
        if False:
            i = 10
            return i + 15
        result = self._both()
        self.assertAllEqual(result, tf.constant([11, 102, 3, 204, 205], dtype=tf.int64))

    @test_util.run_in_graph_and_eager_modes
    def test_inconsistent_inputs_error(self):
        if False:
            return 10
        a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
        a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
        a = [a1, a2]
        b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
        cond = tf.constant([False, False, True, False, False], dtype=bool)
        with self.assertRaisesRegex((errors_impl.InvalidArgumentError, ValueError), '(a_values\\[0\\] and b_values must have the same shape)|(Shapes must be equal rank, but are 2 and 1)'):
            self.evaluate(multiplex_4_op.multiplex(cond, a, b))
if __name__ == '__main__':
    tf.test.main()