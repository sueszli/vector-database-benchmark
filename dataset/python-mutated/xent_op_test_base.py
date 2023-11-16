"""Tests for SoftmaxCrossEntropyWithLogits op."""
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad
from tensorflow.python.platform import test

class XentOpTestBase(test.TestCase):

    def _opFwdBwd(self, labels, logits, axis=-1):
        if False:
            for i in range(10):
                print('nop')
        ' Runs the op-under-test both forwards and backwards.'
        logits = ops.convert_to_tensor(logits)
        with backprop.GradientTape() as tape:
            tape.watch(logits)
            loss = nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits, dim=axis)
        return (loss, tape.gradient(loss, logits))

    def _npXent(self, labels, logits, dim=-1):
        if False:
            print('Hello World!')
        if dim == -1:
            dim = len(logits.shape) - 1
        one_only_on_dim = list(logits.shape)
        one_only_on_dim[dim] = 1
        e = np.exp(logits - np.reshape(np.amax(logits, axis=dim), one_only_on_dim))
        probs = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
        bp = probs - labels
        l = -np.sum(labels * np.log(probs + 1e-20), axis=dim)
        return (l, bp)

    def _testXent2D(self, np_labels, np_logits, with_placeholders=False, expected_gradient=None):
        if False:
            i = 10
            return i + 15
        (np_loss, np_gradient) = self._npXent(labels=np_labels, logits=np_logits)
        if expected_gradient is not None:
            np_gradient = expected_gradient
        with self.cached_session() as sess:
            if with_placeholders:
                logits_placeholder = array_ops.placeholder(np_logits.dtype)
                labels_placeholder = array_ops.placeholder(np_labels.dtype)
                (loss, gradient) = self._opFwdBwd(labels_placeholder, logits_placeholder)
                (tf_loss, tf_gradient) = sess.run([loss, gradient], feed_dict={labels_placeholder: np_labels, logits_placeholder: np_logits})
            else:
                (loss, gradient) = self._opFwdBwd(np_labels, np_logits)
                (tf_loss, tf_gradient) = self.evaluate([loss, gradient])
        self.assertAllCloseAccordingToType(np_loss, tf_loss, half_rtol=0.01)
        self.assertAllCloseAccordingToType(np_gradient, tf_gradient)

    def _testXentND(self, np_labels, np_logits, dim=-1):
        if False:
            for i in range(10):
                print('nop')
        (np_loss, _) = self._npXent(np_labels, np_logits, dim=dim)
        loss = nn_ops.softmax_cross_entropy_with_logits(labels=np_labels, logits=np_logits, dim=dim)
        tf_loss = self.evaluate(loss)
        self.assertAllCloseAccordingToType(np_loss, tf_loss)

    def _testSingleClass(self, expected_gradient=[[2.0], [1.0], [0.0], [0.0]]):
        if False:
            while True:
                i = 10
        for dtype in (np.float16, np.float32, dtypes.bfloat16.as_numpy_dtype):
            (loss, gradient) = self._opFwdBwd(labels=np.array([[-1.0], [0.0], [1.0], [1.0]]).astype(dtype), logits=np.array([[1.0], [-1.0], [0.0], [1.0]]).astype(dtype))
            self.assertAllClose([0.0, 0.0, 0.0, 0.0], loss)
            self.assertAllClose(expected_gradient, gradient)

    def testSingleClass(self):
        if False:
            while True:
                i = 10
        'This method is structured to be easily overridden by a child class.'
        self._testSingleClass()

    def testNpXent(self):
        if False:
            for i in range(10):
                print('nop')
        logits = [[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]
        labels = [[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 0.5, 0.0]]
        (np_loss, np_gradient) = self._npXent(np.array(labels), np.array(logits))
        self.assertAllClose(np.array([[0.25, 0.25, 0.25, -0.75], [0.0321, -0.4129, -0.2632, 0.6439]]), np_gradient, rtol=0.001, atol=0.001)
        self.assertAllClose(np.array([1.3862, 1.9401]), np_loss, rtol=0.001, atol=0.001)

    @test_util.run_deprecated_v1
    def _testLabelsBroadcast(self, uniform_labels_gradient):
        if False:
            for i in range(10):
                print('nop')
        labels = np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float16)
        logits = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(np.float16)
        self._testXent2D(labels, logits, with_placeholders=True)
        labels = np.array([[1.0]]).astype(np.float16)
        logits = np.array([[1.0], [2.0]]).astype(np.float16)
        self._testXent2D(labels, logits, with_placeholders=True)
        labels = np.array([[0.0], [2.0], [0.25]]).astype(np.float16)
        logits = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]).astype(np.float16)
        self._testXent2D(labels, logits, with_placeholders=True, expected_gradient=uniform_labels_gradient)

    def testLabelsBroadcast(self):
        if False:
            for i in range(10):
                print('nop')
        'This method is structured to be easily overridden by a child class.'
        self._testLabelsBroadcast(uniform_labels_gradient=[[0.25, 0.25, 0.25, 0.25], [-1.968, -1.913, -1.763, -1.355], [-0.218, -0.163, -0.013, 0.394]])

    @test_util.run_deprecated_v1
    def testShapeMismatch(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            with self.assertRaises(ValueError):
                self._opFwdBwd(labels=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], logits=[[0.0, 1.0], [2.0, 3.0]])

    def testHalf(self):
        if False:
            i = 10
            return i + 15
        labels = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 0.5, 0.0]]).astype(np.float16)
        logits = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(np.float16)
        self._testXent2D(labels, logits)

    def testBfloat16(self):
        if False:
            for i in range(10):
                print('nop')
        labels = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 0.5, 0.0]]).astype(dtypes.bfloat16.as_numpy_dtype)
        logits = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(dtypes.bfloat16.as_numpy_dtype)
        self._testXent2D(labels, logits)

    def testFloat(self):
        if False:
            i = 10
            return i + 15
        labels = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 0.5, 0.0]]).astype(np.float32)
        logits = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(np.float32)
        self._testXent2D(labels, logits)

    def testDouble(self):
        if False:
            for i in range(10):
                print('nop')
        labels = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 0.5, 0.0]]).astype(np.float64)
        logits = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(np.float64)
        self._testXent2D(labels, logits)

    @test_util.run_deprecated_v1
    def testGradient(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            labels = constant_op.constant([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5], shape=[3, 4], dtype=dtypes.float64, name='labels')
            logits = constant_op.constant([0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4], shape=[3, 4], dtype=dtypes.float64, name='logits')
            x = nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xent')
            err = gradient_checker.compute_gradient_error(logits, [3, 4], x, [3])
            op_names = [op.op_def.name for op in sess.graph.get_operations() if op.op_def]
            self.assertNotIn('BatchMatMul', op_names)
            self.assertNotIn('BatchMatMulV2', op_names)
        self.assertLess(err, 5e-08)

    @test_util.run_deprecated_v1
    def testGradientLabelWithV2(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            labels = constant_op.constant([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5], shape=[3, 4], dtype=dtypes.float64, name='labels')
            logits = constant_op.constant([0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4], shape=[3, 4], dtype=dtypes.float64, name='logits')
            x = nn_ops.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name='xent')
            err = gradient_checker.compute_gradient_error(labels, [3, 4], x, [3])
        self.assertLess(err, 5e-08)

    @test_util.run_deprecated_v1
    def testSecondGradient(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            labels = constant_op.constant([0.0, 0.0, 1.0 / 3, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.5 / 3, 0.0, 0.5 / 3], shape=[12], dtype=dtypes.float64, name='labels')
            logits = constant_op.constant([0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4], shape=[12], dtype=dtypes.float64, name='logits')
            x = nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xent')
            loss = math_ops.reduce_sum(x)
            gradients = gradients_impl.gradients(loss, [logits])[0]
            err = gradient_checker.compute_gradient_error(logits, [12], gradients, [12])
            if not config.is_op_determinism_enabled():
                op_names = [op.op_def.name for op in sess.graph.get_operations() if op.op_def]
                self.assertIn('BatchMatMulV2', op_names)
        self.assertLess(err, 5e-08)

    def test3D(self):
        if False:
            return 10
        labels = np.array([[[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]], [[0.0, 0.5, 0.5, 0.0], [0.5, 0.5, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]).astype(np.float32)
        logits = np.array([[[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]], [[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]], [[5.0, 4.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0]]]).astype(np.float32)
        self._testXentND(labels, logits, dim=0)
        self._testXentND(labels, logits, dim=1)
        self._testXentND(labels, logits, dim=-1)

    def testZeroDimension(self):
        if False:
            for i in range(10):
                print('nop')
        labels = np.zeros([0, 2, 4]).astype(np.float32)
        logits = np.zeros([0, 2, 4]).astype(np.float32)
        (np_loss, _) = self._npXent(labels=labels, logits=logits)
        loss = nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        tf_loss = self.evaluate(loss)
        self.assertAllEqual(np_loss, tf_loss)