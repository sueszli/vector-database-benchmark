"""Tests for mfcc_ops."""
from absl.testing import parameterized
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.signal import mfcc_ops
from tensorflow.python.platform import test

@test_util.run_all_in_graph_and_eager_modes
class MFCCTest(test.TestCase, parameterized.TestCase):

    def test_error(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            signal = array_ops.zeros((2, 3, 0))
            mfcc_ops.mfccs_from_log_mel_spectrograms(signal)

    @parameterized.parameters(dtypes.float32, dtypes.float64)
    def test_basic(self, dtype):
        if False:
            print('Hello World!')
        'A basic test that the op runs on random input.'
        signal = random_ops.random_normal((2, 3, 5), dtype=dtype)
        self.evaluate(mfcc_ops.mfccs_from_log_mel_spectrograms(signal))

    def test_unknown_shape(self):
        if False:
            return 10
        'A test that the op runs when shape and rank are unknown.'
        if context.executing_eagerly():
            return
        signal = array_ops.placeholder_with_default(random_ops.random_normal((2, 3, 5)), tensor_shape.TensorShape(None))
        self.assertIsNone(signal.shape.ndims)
        self.evaluate(mfcc_ops.mfccs_from_log_mel_spectrograms(signal))
if __name__ == '__main__':
    test.main()