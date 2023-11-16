"""Tests for maximum spanning tree ops."""
import math
import numpy as np
import tensorflow as tf
from dragnn.python import mst_ops

class MstOpsTest(tf.test.TestCase):
    """Testing rig."""

    def testMaximumSpanningTree(self):
        if False:
            while True:
                i = 10
        'Tests that the MST op can recover a simple tree.'
        with self.test_session() as session:
            num_nodes = tf.constant([4, 3], tf.int32)
            scores = tf.constant([[[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 4]], [[4, 3, 2, 9], [0, 0, 2, 9], [0, 0, 0, 9], [9, 9, 9, 9]]], tf.int32)
            mst_outputs = mst_ops.maximum_spanning_tree(num_nodes, scores, forest=False)
            (max_scores, argmax_sources) = session.run(mst_outputs)
            tf.logging.info('\nmax_scores=%s\nargmax_sources=\n%s', max_scores, argmax_sources)
            self.assertAllEqual(max_scores, [7, 6])
            self.assertAllEqual(argmax_sources, [[3, 0, 1, 3], [0, 2, 0, -1]])

    def testMaximumSpanningTreeGradient(self):
        if False:
            return 10
        'Tests the MST max score gradient.'
        with self.test_session() as session:
            num_nodes = tf.constant([4, 3], tf.int32)
            scores = tf.constant([[[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 4]], [[4, 3, 2, 9], [0, 0, 2, 9], [0, 0, 0, 9], [9, 9, 9, 9]]], tf.int32)
            mst_ops.maximum_spanning_tree(num_nodes, scores, forest=False, name='MST')
            mst_op = session.graph.get_operation_by_name('MST')
            d_loss_d_max_scores = tf.constant([3, 7], tf.float32)
            (d_loss_d_num_nodes, d_loss_d_scores) = mst_ops.maximum_spanning_tree_gradient(mst_op, d_loss_d_max_scores)
            self.assertTrue(d_loss_d_num_nodes is None)
            tf.logging.info('\nd_loss_d_scores=\n%s', d_loss_d_scores.eval())
            self.assertAllEqual(d_loss_d_scores.eval(), [[[0, 0, 0, 3], [3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 3]], [[7, 0, 0, 0], [0, 0, 7, 0], [7, 0, 0, 0], [0, 0, 0, 0]]])

    def testMaximumSpanningTreeGradientError(self):
        if False:
            print('Hello World!')
        'Numerically validates the max score gradient.'
        with self.test_session():
            scores_raw = [[[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 4]], [[4, 3, 2, 9], [0, 0, 2, 9], [0, 0, 0, 9], [9, 9, 9, 9]]]
            scores = tf.constant(scores_raw, tf.float64)
            init_scores = np.array(scores_raw)
            num_nodes = tf.constant([4, 3], tf.int32)
            max_scores = mst_ops.maximum_spanning_tree(num_nodes, scores, forest=False)[0]
            gradient_error = tf.test.compute_gradient_error(scores, [2, 4, 4], max_scores, [2], init_scores)
            tf.logging.info('gradient_error=%s', gradient_error)

    def testLogPartitionFunctionOneTree(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the log partition function with one feasible tree with score 1.'
        with self.test_session():
            for forest in [False, True]:
                pad = 12345.6
                scores = tf.constant([[[1, pad, pad], [pad, pad, pad], [pad, pad, pad]], [[1, 0, pad], [1, 0, pad], [pad, pad, pad]], [[1, 0, 0], [1, 0, 0], [0, 1, 0]]], tf.float64)
                scores = tf.log(scores)
                num_nodes = tf.constant([1, 2, 3], tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest)
                self.assertAlmostEqual(tf.exp(log_partition_functions[0]).eval(), 1.0)
                self.assertAlmostEqual(tf.exp(log_partition_functions[1]).eval(), 1.0)
                self.assertAlmostEqual(tf.exp(log_partition_functions[2]).eval(), 1.0)

    def testLogPartitionFunctionOneTreeScaled(self):
        if False:
            print('Hello World!')
        'Tests the log partition function with one feasible tree.'
        with self.test_session():
            for forest in [False, True]:
                pad = 12345.6
                scores = tf.constant([[[2, pad, pad], [pad, pad, pad], [pad, pad, pad]], [[3, 0, pad], [5, 0, pad], [pad, pad, pad]], [[7, 0, 0], [11, 0, 0], [0, 13, 0]]], tf.float64)
                scores = tf.log(scores)
                num_nodes = tf.constant([1, 2, 3], tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest)
                self.assertAlmostEqual(tf.exp(log_partition_functions[0]).eval(), 2.0)
                self.assertAlmostEqual(tf.exp(log_partition_functions[1]).eval(), 3.0 * 5.0)
                self.assertAlmostEqual(tf.exp(log_partition_functions[2]).eval(), 7.0 * 11.0 * 13.0)

    def testLogPartitionFunctionTwoTreesScaled(self):
        if False:
            print('Hello World!')
        'Tests the log partition function with two feasible trees.'
        with self.test_session():
            for forest in [False, True]:
                pad = 12345.6
                scores = tf.constant([[[2, 0, 0, pad], [3, 0, 0, pad], [5, 7, 0, pad], [pad, pad, pad, pad]], [[0, 11, 0, 13], [0, 17, 0, 0], [0, 19, 0, 0], [0, 23, 0, 0]]], tf.float64)
                scores = tf.log(scores)
                num_nodes = tf.constant([3, 4], tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest)
                self.assertAlmostEqual(tf.exp(log_partition_functions[0]).eval(), 2.0 * 3.0 * 5.0 + 2.0 * 3.0 * 7.0)
                self.assertAlmostEqual(tf.exp(log_partition_functions[1]).eval(), 11.0 * 17.0 * 19.0 * 23.0 + 13.0 * 17.0 * 19.0 * 23.0)

    def testLogPartitionFunctionInfeasible(self):
        if False:
            print('Hello World!')
        'Tests the log partition function on infeasible scores.'
        with self.test_session():
            for forest in [False, True]:
                pad = 12345.6
                scores = tf.constant([[[0, 1, pad, pad], [1, 0, pad, pad], [pad, pad, pad, pad], [pad, pad, pad, pad]], [[0, 1, 0, pad], [0, 0, 1, pad], [1, 0, 0, pad], [pad, pad, pad, pad]], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]], tf.float64)
                scores = tf.log(scores)
                num_nodes = tf.constant([2, 3, 4], tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest)
                self.assertAlmostEqual(tf.exp(log_partition_functions[0]).eval(), 0.0)
                self.assertAlmostEqual(tf.exp(log_partition_functions[1]).eval(), 0.0)
                self.assertAlmostEqual(tf.exp(log_partition_functions[2]).eval(), 0.0)

    def testLogPartitionFunctionAllTrees(self):
        if False:
            while True:
                i = 10
        'Tests the log partition function with all trees feasible.'
        with self.test_session():
            for forest in [False, True]:
                scores = tf.zeros([10, 10, 10], tf.float64)
                num_nodes = tf.range(1, 11, dtype=tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest)
                base_offset = 1 if forest else 0
                for size in range(1, 11):
                    self.assertAlmostEqual(log_partition_functions[size - 1].eval(), (size - 1) * math.log(size + base_offset))

    def testLogPartitionFunctionWithVeryHighValues(self):
        if False:
            print('Hello World!')
        'Tests the overflow protection in the log partition function.'
        with self.test_session():
            for forest in [False, True]:
                scores = 1000 * tf.ones([10, 10, 10], tf.float64)
                num_nodes = tf.range(1, 11, dtype=tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest)
                base_offset = 1 if forest else 0
                for size in range(1, 11):
                    self.assertAlmostEqual(log_partition_functions[size - 1].eval(), (size - 1) * math.log(size + base_offset) + size * 1000)

    def testLogPartitionFunctionWithVeryLowValues(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the underflow protection in the log partition function.'
        with self.test_session():
            for forest in [False, True]:
                scores = -1000 * tf.ones([10, 10, 10], tf.float64)
                num_nodes = tf.range(1, 11, dtype=tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest)
                base_offset = 1 if forest else 0
                for size in range(1, 11):
                    self.assertAlmostEqual(log_partition_functions[size - 1].eval(), (size - 1) * math.log(size + base_offset) - size * 1000)

    def testLogPartitionFunctionGradientError(self):
        if False:
            i = 10
            return i + 15
        'Validates the log partition function gradient.'
        with self.test_session():
            for forest in [False, True]:
                scores_raw = [[[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 4]], [[4, 3, 2, 9], [0, 0, 2, 9], [0, 0, 0, 9], [9, 9, 9, 9]]]
                scores = tf.constant(scores_raw, tf.float64)
                init_scores = np.array(scores_raw)
                num_nodes = tf.constant([4, 3], tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest)
                gradient_error = tf.test.compute_gradient_error(scores, [2, 4, 4], log_partition_functions, [2], init_scores)
                tf.logging.info('forest=%s gradient_error=%s', forest, gradient_error)
                self.assertLessEqual(gradient_error, 1e-07)

    def testLogPartitionFunctionGradientErrorFailsIfInfeasible(self):
        if False:
            print('Hello World!')
        'Tests that the partition function gradient fails on infeasible scores.'
        with self.test_session():
            for forest in [False, True]:
                pad = 12345.6
                scores_raw = [[[0, 1, pad, pad], [1, 0, pad, pad], [pad, pad, pad, pad], [pad, pad, pad, pad]], [[0, 1, 0, pad], [0, 0, 1, pad], [1, 0, 0, pad], [pad, pad, pad, pad]], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]]
                scores = tf.log(scores_raw)
                init_scores = np.log(np.array(scores_raw))
                num_nodes = tf.constant([2, 3, 4], tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest)
                with self.assertRaises(Exception):
                    tf.test.compute_gradient_error(scores, [3, 4, 4], log_partition_functions, [3], init_scores)

    def testLogPartitionFunctionGradientErrorOkIfInfeasibleWithClipping(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the log partition function gradient is OK after clipping.'
        with self.test_session():
            for forest in [False, True]:
                pad = 12345.6
                scores_raw = [[[0, 1, pad, pad], [1, 0, pad, pad], [pad, pad, pad, pad], [pad, pad, pad, pad]], [[0, 1, 0, pad], [0, 0, 1, pad], [1, 0, 0, pad], [pad, pad, pad, pad]], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]]
                scores = tf.log(scores_raw)
                init_scores = np.log(np.array(scores_raw))
                num_nodes = tf.constant([2, 3, 4], tf.int32)
                log_partition_functions = mst_ops.log_partition_function(num_nodes, scores, forest=forest, max_dynamic_range=10)
                gradient_error = tf.test.compute_gradient_error(scores, [3, 4, 4], log_partition_functions, [3], init_scores)
                tf.logging.info('forest=%s gradient_error=%s', forest, gradient_error)
                self.assertLessEqual(gradient_error, 0.001)
if __name__ == '__main__':
    tf.test.main()