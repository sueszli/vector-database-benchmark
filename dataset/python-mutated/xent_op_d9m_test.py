"""Tests for deterministic functionality of SoftmaxCrossEntropyWithLogits op."""
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.nn_ops import xent_op_test_base
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad
from tensorflow.python.platform import test

class XentOpDeterminismExceptionsTest(test.TestCase):
    """Test d9m-unimplemented exceptions from SoftmaxXentWithLogitsOp.

  Test that tf.errors.UnimplementedError is thrown, as appropriate, by the GPU
  code-paths through SoftmaxXentWithLogitsOp when deterministic ops are
  enabled.

  This test assumes that xent_op_test.py runs equivalent test cases when
  deterministic ops are not enabled and will therefore detect erroneous
  exception throwing in those cases.
  """

    @test_util.run_gpu_only
    @test_util.run_in_graph_and_eager_modes
    def testExceptionThrowing(self):
        if False:
            i = 10
            return i + 15
        with self.session(), test_util.force_gpu():
            for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
                features = constant_op.constant([[0.3, 0.5], [0.5, 0.6]], dtype=dtype)
                labels = constant_op.constant([[0.2, 0.4], [0.1, 0.2]], dtype=dtype)
                with self.assertRaisesRegex(errors_impl.UnimplementedError, 'The GPU implementation of SoftmaxCrossEntropyWithLogits that ' + 'would have been executed is not deterministic. Note that the ' + 'Python API uses an alternative, deterministic, GPU-accelerated ' + 'path when determinism is enabled.'):
                    result = gen_nn_ops.softmax_cross_entropy_with_logits(features=features, labels=labels)
                    self.evaluate(result)

class XentOpDeterministicTest(xent_op_test_base.XentOpTestBase):
    """Test that SoftmaxCrossEntropyWithLogits operates reproducibly.

  Inheriting from xent_op_test_base.XentTestBase ensures that regular op
  functionality is correct when the deterministic code-path is selected.

  Note that because nn_ops.softmax_cross_entropy_with_logits calls
  nn_ops.cross_entropy_with_logits_v2, the focus of testing is on the
  former in order to test both.
  """

    def _randomFloats(self, shape, dtype, normalized_rows=False):
        if False:
            return 10
        a = (2 * np.random.random_sample(shape) - 1).astype(dtype)
        if normalized_rows:

            def normalize(row):
                if False:
                    return 10
                return row / row.sum()
            a = np.apply_along_axis(normalize, 1, a)
        return constant_op.constant(a)

    def _generateInputs(self, dtype, seed=123, forward_not_backward=False):
        if False:
            return 10
        batch_size = 1024
        if forward_not_backward and dtype == np.float16:
            classes_count = 20000
        else:
            classes_count = 3000
        shape = (batch_size, classes_count)
        np.random.seed(seed)
        labels = self._randomFloats(shape, dtype, normalized_rows=True)
        logits = self._randomFloats(shape, dtype)
        return (labels, logits)

    @test_util.run_in_graph_and_eager_modes
    def testForward(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            for dtype in [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]:
                for trial in range(5):
                    seed = 123 + trial
                    (labels, logits) = self._generateInputs(dtype, seed=seed, forward_not_backward=True)
                    result_a = nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                    result_b = nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                    self.assertAllEqual(result_a, result_b)

    @test_util.run_in_graph_and_eager_modes
    def testBackward(self):
        if False:
            return 10
        with self.cached_session():
            for dtype in [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]:
                (labels, logits) = self._generateInputs(dtype, seed=456)
                output_shape = labels.shape[0]

                def gradients(seed):
                    if False:
                        return 10
                    np.random.seed(seed)
                    upstream_gradients = self._randomFloats(output_shape, dtype)
                    with backprop.GradientTape(persistent=True) as tape:
                        tape.watch(labels)
                        tape.watch(logits)
                        op_output = nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                        gradient_injector_output = op_output * upstream_gradients
                    return tape.gradient(gradient_injector_output, [labels, logits])
                for trial in range(5):
                    seed = 456 + trial
                    (labels_grad_a, logits_grad_a) = gradients(seed=seed)
                    (labels_grad_b, logits_grad_b) = gradients(seed=seed)
                    self.assertAllEqual(labels_grad_a, labels_grad_b)
                    self.assertAllEqual(logits_grad_a, logits_grad_b)

    def testSingleClass(self):
        if False:
            print('Hello World!')
        'Modify testing of gradient for single-class case.\n\n    The deterministic implementation does not produce the gradients expected by\n    the original test (for the nondeterministic functionality) when the labels\n    vector is not a valid probability distribution.\n\n    labels: [[-1.], [0.], [1.], [1.]]\n    logits: [[1.], [-1.], [0.], [1.]]\n\n                   nondeterministic               deterministic\n    dloss/dlogits: [[2.0], [1.0], [0.0], [0.0]]   [[0.0], [0.0], [0.0], [0.0]]\n\n    Note that only the second two label vectors are valid probability\n    distributions (as required by the API) and that the gradient matches for\n    those cases.\n\n    TODO(duncanriach): Further investigate the source of the difference in\n                       the gradients for this case.\n    '
        self._testSingleClass(expected_gradient=[[0.0], [0.0], [0.0], [0.0]])

    def testLabelsBroadcast(self):
        if False:
            print('Hello World!')
        'Modify testing of gradient for labels-broadcast case.\n\n    The deterministic implementation does not produce the gradients expected by\n    the original test (for the nondeterministic functionality) when the labels\n    vector (after broadcasting) is not a valid probability distribution.\n\n    labels: [[0.], [2.], [0.25]]\n    logits: [[1., 1., 1., 1.],\n             [1., 2., 3., 4.],\n             [1., 2., 3., 4.]]\n\n    dloss/dlogits (nondeterministic):\n        [[ 0.25 ,  0.25 ,  0.25 ,  0.25 ],\n         [-1.968, -1.913, -1.763, -1.355],\n         [-0.218, -0.163, -0.013,  0.394]]\n\n    dloss/dlogits (determinsitic):\n        [[ 0.   ,  0.   ,  0.   ,  0.   ],\n         [-1.743, -1.303, -0.105,  3.150],\n         [-0.218, -0.163, -0.013,  0.394]]\n\n    Note that neither of the first two broadcast label vectors is a valid\n    probability distribution (as required by the API) and that these are the\n    cases that yield different gradients for nondeterministic vs determinsitic\n    implementations.\n\n    TODO(duncanriach): Further investigate the source of the difference in\n                       the gradient for this case.\n    '
        self._testLabelsBroadcast(uniform_labels_gradient=[[0.0, 0.0, 0.0, 0.0], [-1.743, -1.303, -0.105, 3.15], [-0.218, -0.163, -0.013, 0.394]])
if __name__ == '__main__':
    config.enable_op_determinism()
    test.main()