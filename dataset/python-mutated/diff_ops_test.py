"""Tests for diff.py."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import math
from tensorflow.python.framework import test_util

class DiffOpsTest(parameterized.TestCase, tf.test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_diffs(self):
        if False:
            return 10
        x = tf.constant([1, 2, 3, 4, 5])
        dx = self.evaluate(math.diff(x, order=1, exclusive=False))
        np.testing.assert_array_equal(dx, [1, 1, 1, 1, 1])
        dx1 = self.evaluate(math.diff(x, order=1, exclusive=True))
        np.testing.assert_array_equal(dx1, [1, 1, 1, 1])
        dx2 = self.evaluate(math.diff(x, order=2, exclusive=False))
        np.testing.assert_array_equal(dx2, [1, 2, 2, 2, 2])

    @test_util.deprecated_graph_mode_only
    def test_diffs_differentiable(self):
        if False:
            while True:
                i = 10
        'Tests that the diffs op is differentiable.'
        x = tf.constant(2.0)
        xv = tf.stack([x, x * x, x * x * x], axis=0)
        dxv = self.evaluate(math.diff(xv))
        np.testing.assert_array_equal(dxv, [2.0, 2.0, 4.0])
        grad = self.evaluate(tf.gradients(math.diff(xv), x)[0])
        self.assertEqual(grad, 12.0)

    @parameterized.named_parameters({'testcase_name': 'exclusive_0', 'exclusive': True, 'axis': 0, 'dx_true': np.array([[9, 18, 27, 36]])}, {'testcase_name': 'exclusive_1', 'exclusive': True, 'axis': 1, 'dx_true': np.array([[1, 1, 1], [10, 10, 10]])}, {'testcase_name': 'nonexclusive_0', 'exclusive': False, 'axis': 0, 'dx_true': np.array([[1, 2, 3, 4], [9, 18, 27, 36]])}, {'testcase_name': 'nonexclusive_1', 'exclusive': False, 'axis': 1, 'dx_true': np.array([[1, 1, 1, 1], [10, 10, 10, 10]])})
    @test_util.run_in_graph_and_eager_modes
    def test_batched_axis(self, exclusive, axis, dx_true):
        if False:
            i = 10
            return i + 15
        'Tests batch diff works with axis argument use of exclusivity.'
        x = tf.constant([[1, 2, 3, 4], [10, 20, 30, 40]])
        dx = self.evaluate(math.diff(x, order=1, exclusive=exclusive, axis=axis))
        self.assertAllEqual(dx, dx_true)
if __name__ == '__main__':
    tf.test.main()