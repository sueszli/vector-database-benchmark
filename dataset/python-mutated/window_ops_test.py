"""Tests for window_ops."""
import functools
import itertools
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.kernel_tests.signal import test_util
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.platform import test
_TF_DTYPE_TOLERANCE = [(dtypes.float16, 0.01), (dtypes.float32, 1e-06), (dtypes.float64, 1e-09)]
_WINDOW_LENGTHS = [1, 2, 3, 4, 5, 31, 64, 128]
_MDCT_WINDOW_LENGTHS = [4, 16, 256]

def _scipy_raised_cosine(length, symmetric=True, a=0.5, b=0.5):
    if False:
        i = 10
        return i + 15
    'A simple implementation of a raised cosine window that matches SciPy.\n\n  https://en.wikipedia.org/wiki/Window_function#Hann_window\n  https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/windows.py#L615\n\n  Args:\n    length: The window length.\n    symmetric: Whether to create a symmetric window.\n    a: The alpha parameter of the raised cosine window.\n    b: The beta parameter of the raised cosine window.\n\n  Returns:\n    A raised cosine window of length `length`.\n  '
    if length == 1:
        return np.ones(1)
    odd = length % 2
    if not symmetric and (not odd):
        length += 1
    window = a - b * np.cos(2.0 * np.pi * np.arange(length) / (length - 1))
    if not symmetric and (not odd):
        window = window[:-1]
    return window

@tf_test_util.run_all_in_graph_and_eager_modes
class WindowOpsTest(test.TestCase, parameterized.TestCase):

    def _check_mdct_window(self, window, tol=1e-06):
        if False:
            print('Hello World!')
        'Check that an MDCT window satisfies necessary conditions.'
        wlen = int(np.shape(window)[0])
        assert wlen % 4 == 0
        half_len = wlen // 2
        squared_sums = window[:half_len] ** 2 + window[half_len:] ** 2
        self.assertAllClose(squared_sums, np.ones((half_len,)), tol, tol)
        sym_diff = window[:half_len] - window[-1:half_len - 1:-1]
        self.assertAllClose(sym_diff, np.zeros((half_len,)), tol, tol)

    def _compare_window_fns(self, np_window_fn, tf_window_fn, window_length, periodic, tf_dtype_tol):
        if False:
            for i in range(10):
                print('nop')
        (tf_dtype, tol) = tf_dtype_tol
        np_dtype = tf_dtype.as_numpy_dtype
        expected = np_window_fn(window_length, symmetric=not periodic).astype(np_dtype)
        actual = tf_window_fn(window_length, periodic=periodic, dtype=tf_dtype)
        self.assertAllClose(expected, actual, tol, tol)

    @parameterized.parameters(itertools.product(_WINDOW_LENGTHS, (4.0, 8.0, 10.0, 12.0), _TF_DTYPE_TOLERANCE))
    def test_kaiser_window(self, window_length, beta, tf_dtype_tol):
        if False:
            i = 10
            return i + 15
        'Check that kaiser_window matches np.kaiser behavior.'
        self.assertAllClose(np.kaiser(window_length, beta), window_ops.kaiser_window(window_length, beta, tf_dtype_tol[0]), tf_dtype_tol[1], tf_dtype_tol[1])

    @parameterized.parameters(itertools.product(_WINDOW_LENGTHS, (False, True), _TF_DTYPE_TOLERANCE))
    def test_hann_window(self, window_length, periodic, tf_dtype_tol):
        if False:
            i = 10
            return i + 15
        'Check that hann_window matches scipy.signal.hann behavior.'
        self._compare_window_fns(functools.partial(_scipy_raised_cosine, a=0.5, b=0.5), window_ops.hann_window, window_length, periodic, tf_dtype_tol)

    @parameterized.parameters(itertools.product(_WINDOW_LENGTHS, (False, True), _TF_DTYPE_TOLERANCE))
    def test_hamming_window(self, window_length, periodic, tf_dtype_tol):
        if False:
            print('Hello World!')
        "Check that hamming_window matches scipy.signal.hamming's behavior."
        self._compare_window_fns(functools.partial(_scipy_raised_cosine, a=0.54, b=0.46), window_ops.hamming_window, window_length, periodic, tf_dtype_tol)

    @parameterized.parameters(itertools.product((window_ops.hann_window, window_ops.hamming_window, window_ops.kaiser_window, window_ops.kaiser_bessel_derived_window, window_ops.vorbis_window), (False, True), _TF_DTYPE_TOLERANCE))
    def test_constant_folding(self, window_fn, periodic, tf_dtype_tol):
        if False:
            return 10
        'Window functions should be constant foldable for constant inputs.'
        if context.executing_eagerly():
            return
        g = ops.Graph()
        with g.as_default():
            try:
                window = window_fn(100, periodic=periodic, dtype=tf_dtype_tol[0])
            except TypeError:
                window = window_fn(100, dtype=tf_dtype_tol[0])
            rewritten_graph = test_util.grappler_optimize(g, [window])
            self.assertLen(rewritten_graph.node, 1)

    @parameterized.parameters((window_ops.hann_window, 10, False, dtypes.float32), (window_ops.hann_window, 10, True, dtypes.float32), (window_ops.hamming_window, 10, False, dtypes.float32), (window_ops.hamming_window, 10, True, dtypes.float32), (window_ops.vorbis_window, 12, None, dtypes.float32))
    def test_tflite_convert(self, window_fn, window_length, periodic, dtype):
        if False:
            print('Hello World!')

        def fn(window_length):
            if False:
                i = 10
                return i + 15
            try:
                return window_fn(window_length, periodic=periodic, dtype=dtype)
            except TypeError:
                return window_fn(window_length, dtype=dtype)
        tflite_model = test_util.tflite_convert(fn, [tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)])
        window_length = np.array(window_length).astype(np.int32)
        (actual_output,) = test_util.evaluate_tflite_model(tflite_model, [window_length])
        expected_output = self.evaluate(fn(window_length))
        self.assertAllClose(actual_output, expected_output, rtol=1e-06, atol=1e-06)

    @parameterized.parameters(itertools.product(_MDCT_WINDOW_LENGTHS, _TF_DTYPE_TOLERANCE))
    def test_vorbis_window(self, window_length, tf_dtype_tol):
        if False:
            for i in range(10):
                print('nop')
        'Check if vorbis windows satisfy MDCT window conditions.'
        self._check_mdct_window(window_ops.vorbis_window(window_length, dtype=tf_dtype_tol[0]), tol=tf_dtype_tol[1])

    @parameterized.parameters(itertools.product(_MDCT_WINDOW_LENGTHS, (4.0, 8.0, 10.0, 12.0), _TF_DTYPE_TOLERANCE))
    def test_kaiser_bessel_derived_window(self, window_length, beta, tf_dtype_tol):
        if False:
            while True:
                i = 10
        'Check if Kaiser-Bessel derived windows satisfy MDCT window conditions.'
        self._check_mdct_window(window_ops.kaiser_bessel_derived_window(window_length, beta=beta, dtype=tf_dtype_tol[0]), tol=tf_dtype_tol[1])
if __name__ == '__main__':
    test.main()