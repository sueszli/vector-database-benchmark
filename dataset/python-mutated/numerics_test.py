"""Tests for tensorflow.ops.numerics."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import numerics
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test

class VerifyTensorAllFiniteTest(test.TestCase):

    def testVerifyTensorAllFiniteSucceeds(self):
        if False:
            return 10
        x_shape = [5, 4]
        x = np.random.random_sample(x_shape).astype(np.float32)
        with test_util.use_gpu():
            t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
            t_verified = numerics.verify_tensor_all_finite(t, 'Input is not a number.')
            self.assertAllClose(x, self.evaluate(t_verified))

    def testVerifyTensorAllFiniteFails(self):
        if False:
            return 10
        x_shape = [5, 4]
        x = np.random.random_sample(x_shape).astype(np.float32)
        my_msg = 'Input is not a number.'
        x[0] = np.nan
        with test_util.use_gpu():
            with self.assertRaisesOpError(my_msg):
                t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
                t_verified = numerics.verify_tensor_all_finite(t, my_msg)
                self.evaluate(t_verified)
        x[0] = np.inf
        with test_util.use_gpu():
            with self.assertRaisesOpError(my_msg):
                t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
                t_verified = numerics.verify_tensor_all_finite(t, my_msg)
                self.evaluate(t_verified)

@test_util.run_v1_only('add_check_numerics_op() is meant to be a v1-only API')
class NumericsTest(test.TestCase):

    def testInf(self):
        if False:
            i = 10
            return i + 15
        with self.session(graph=ops.Graph()):
            t1 = constant_op.constant(1.0)
            t2 = constant_op.constant(0.0)
            a = math_ops.div(t1, t2)
            check = numerics.add_check_numerics_ops()
            a = control_flow_ops.with_dependencies([check], a)
            with self.assertRaisesOpError('Inf'):
                self.evaluate(a)

    def testNaN(self):
        if False:
            while True:
                i = 10
        with self.session(graph=ops.Graph()):
            t1 = constant_op.constant(0.0)
            t2 = constant_op.constant(0.0)
            a = math_ops.div(t1, t2)
            check = numerics.add_check_numerics_ops()
            a = control_flow_ops.with_dependencies([check], a)
            with self.assertRaisesOpError('NaN'):
                self.evaluate(a)

    def testBoth(self):
        if False:
            while True:
                i = 10
        with self.session(graph=ops.Graph()):
            t1 = constant_op.constant([1.0, 0.0])
            t2 = constant_op.constant([0.0, 0.0])
            a = math_ops.div(t1, t2)
            check = numerics.add_check_numerics_ops()
            a = control_flow_ops.with_dependencies([check], a)
            with self.assertRaisesOpError('Inf and NaN'):
                self.evaluate(a)

    def testPassThrough(self):
        if False:
            print('Hello World!')
        with self.session(graph=ops.Graph()):
            t1 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            checked = array_ops.check_numerics(t1, message='pass through test')
            value = self.evaluate(checked)
            self.assertAllEqual(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), value)
            self.assertEqual([2, 3], checked.get_shape())

    def testControlFlowCond(self):
        if False:
            return 10
        predicate = array_ops.placeholder(dtypes.bool, shape=[])
        _ = cond.cond(predicate, lambda : constant_op.constant([37.0]), lambda : constant_op.constant([42.0]))
        with self.assertRaisesRegex(ValueError, '`tf\\.add_check_numerics_ops\\(\\) is not compatible with TensorFlow control flow operations such as `tf\\.cond\\(\\)` or `tf.while_loop\\(\\)`\\.'):
            numerics.add_check_numerics_ops()

    def testControlFlowWhile(self):
        if False:
            print('Hello World!')
        predicate = array_ops.placeholder(dtypes.bool, shape=[])
        _ = while_loop.while_loop(lambda _: predicate, lambda _: constant_op.constant([37.0]), [constant_op.constant([42.0])])
        with self.assertRaisesRegex(ValueError, '`tf\\.add_check_numerics_ops\\(\\) is not compatible with TensorFlow control flow operations such as `tf\\.cond\\(\\)` or `tf.while_loop\\(\\)`\\.'):
            numerics.add_check_numerics_ops()
if __name__ == '__main__':
    test.main()