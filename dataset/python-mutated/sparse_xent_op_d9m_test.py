"""Tests for deterministic functionality of SparseSoftmaxCrossEntropyWithLogits op."""
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.sparse_ops import sparse_xent_op_test_base
from tensorflow.python.ops.nn_grad import _SparseSoftmaxCrossEntropyWithLogitsGrad
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

class SparseXentOpDeterminismExceptionsTest(test.TestCase):
    """Test d9m-unimplemented exceptions from SparseSoftmaxXentWithLogitsOp.

  Test that tf.errors.UnimplementedError is thrown, as
  appropriate, by the GPU code-paths through SparseSoftmaxXentWithLogitsOp when
  deterministic ops are enabled.

  This test assumes that sparse_xent_op_test.py runs equivalent test cases
  when deterministic ops are not enabled and will therefore detect erroneous
  exception throwing in those cases.
  """

    @test_util.run_gpu_only
    @test_util.run_in_graph_and_eager_modes
    def testExceptionThrowing(self):
        if False:
            print('Hello World!')
        with self.session(), test_util.force_gpu():
            for features_dtype in [dtypes.float16, dtypes.float32]:
                for labels_dtype in [dtypes.int32, dtypes.int64]:
                    features = constant_op.constant([[0.3, 0.5], [0.2, 0.6]], dtype=features_dtype)
                    labels = constant_op.constant([1, 0], dtype=labels_dtype)
                    with self.assertRaisesRegex(errors_impl.UnimplementedError, 'The GPU implementation of SparseSoftmaxCrossEntropyWithLogits ' + 'that would have been executed is not deterministic. Note that ' + 'the Python API uses an alternative, deterministic, ' + 'GPU-accelerated path when determinsim is enabled.'):
                        result = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(features=features, labels=labels)
                        self.evaluate(result)

class SparseXentOpDeterministicTest(sparse_xent_op_test_base.SparseXentOpTestBase):
    """Test that SparseSoftmaxCrossEntropyWithLogits operates reproducibly.

  Inheriting from sparse_xent_op_test_base.SparseXentOpTestBase ensures that
  regular op functionality is correct when the deterministic code-path is
  selected.

  Note that because nn_ops.sparse_softmax_cross_entropy_with_logits_v2 calls
  nn_ops.sparse_softmax_cross_entropy_with_logits directly, the focus of
  testing is on the former in order to test both.
  """

    def _randomInts(self, shape, high, dtype):
        if False:
            for i in range(10):
                print('nop')
        return constant_op.constant(np.random.randint(low=0, high=high, size=shape).astype(dtype))

    def _randomFloats(self, shape, dtype):
        if False:
            return 10
        return constant_op.constant((2 * np.random.random_sample(shape) - 1).astype(dtype))

    def _generateInputs(self, labels_dtype, logits_dtype, seed):
        if False:
            i = 10
            return i + 15
        batch_size = 1024
        classes_count = 1000
        np.random.seed(seed)
        labels_shape = batch_size
        labels = self._randomInts(labels_shape, high=classes_count, dtype=labels_dtype)
        logits_shape = (batch_size, classes_count)
        logits = self._randomFloats(logits_shape, logits_dtype)
        return (labels, logits)

    @test_util.run_in_graph_and_eager_modes
    def testForward(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            for logits_dtype in [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]:
                for labels_dtype in [np.int32, np.int64]:
                    for trial in range(5):
                        seed = 123 + trial
                        (labels, logits) = self._generateInputs(labels_dtype, logits_dtype, seed=seed)
                        result_a = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
                        result_b = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
                        self.assertAllEqual(result_a, result_b)

    @test_util.run_in_graph_and_eager_modes
    def testBackward(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            for logits_dtype in [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]:
                for labels_dtype in [np.int32, np.int64]:
                    (labels, logits) = self._generateInputs(labels_dtype, logits_dtype, seed=456)
                    output_shape = labels.shape[0]

                    def gradients(seed):
                        if False:
                            print('Hello World!')
                        np.random.seed(seed)
                        upstream_gradients = self._randomFloats(output_shape, logits_dtype)
                        with backprop.GradientTape(persistent=True) as tape:
                            tape.watch(logits)
                            op_output = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
                            gradient_injector_output = op_output * upstream_gradients
                        return tape.gradient(gradient_injector_output, logits)
                    for trial in range(5):
                        seed = 456 + trial
                        result_a = gradients(seed=seed)
                        result_b = gradients(seed=seed)
                        self.assertAllEqual(result_a, result_b)

    def testInvalidLabelGPU(self):
        if False:
            while True:
                i = 10
        'Modified test for invalid labels on GPU.\n\n    When running on GPU, the pre-existing, nondeterministic implementation\n    produces NaN (in both the forward and backward directions) for results\n    associated with invalid labels (less than zero or greater than the number of\n    classes minus one). However, while the deterministic implementation also\n    produces NaN in the forward direction, it produces zeros in the backward\n    direction.\n    '
        self._testInvalidLabelGPU(invalid_label_gradient=0.0)

    def testInvalidLabelCPU(self):
        if False:
            for i in range(10):
                print('nop')
        'Modified test for invalid labels on CPU.\n\n    When running on CPU, the pre-existing, nondeterministic implementation\n    throws a custom exception when any of the label values are invalid (less\n    than zero or greater than the number of classes minus one). However, in the\n    deterministic implementation, tf.gather throws an exception instead.\n    '
        self._testInvalidLabelCPU(expected_regex='indices\\[0\\] = 4 is not in \\[0, 4\\)')

    def testLabelsPlaceholderScalar(self):
        if False:
            while True:
                i = 10
        'Test exception-throwing for non-statically-shaped, zero-rank labels.\n\n    The deterministic implementation cannot check for this case because it does\n    not have a specific implementation of SparseSoftmaxXentWithLogitsOp.\n    Instead tf.gather, which is used to create the deterministic implementation,\n    throws an error.\n    '
        self._testLabelsPlaceholderScalar(expected_error_message='Expected batch_dims in the range \\[0, 0\\], ' + 'but got 1')

    def testScalarHandling(self):
        if False:
            i = 10
            return i + 15
        'Test exception-throwing for non-statically-shaped, zero-rank labels.\n\n    The deterministic implementation cannot check for this case because it does\n    not have a specific implementation of SparseSoftmaxXentWithLogitsOp.\n    Instead tf.gather, which is used to create the deterministic implementation,\n    throws an error.\n    '
        self._testScalarHandling(expected_regex='Expected batch_dims in the range \\[0, 0\\], but got 1.*')
if __name__ == '__main__':
    config.enable_op_determinism()
    test.main()