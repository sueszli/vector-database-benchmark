"""Tests for tensorflow.ops.tf.norm."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test as test_lib

def _AddTest(test, test_name, fn):
    if False:
        print('Hello World!')
    test_name = '_'.join(['test', test_name])
    if hasattr(test, test_name):
        raise RuntimeError('Test %s defined more than once' % test_name)
    setattr(test, test_name, fn)

class NormOpTest(test_lib.TestCase):

    @test_util.run_v1_only('b/120545219')
    def testBadOrder(self):
        if False:
            for i in range(10):
                print('nop')
        matrix = [[0.0, 1.0], [2.0, 3.0]]
        for ord_ in ('fro', -7, -1.1, 0):
            with self.assertRaisesRegex(ValueError, "'ord' must be a supported vector norm"):
                linalg_ops.norm(matrix, ord=ord_)
        for ord_ in ('fro', -7, -1.1, 0):
            with self.assertRaisesRegex(ValueError, "'ord' must be a supported vector norm"):
                linalg_ops.norm(matrix, ord=ord_, axis=-1)
        for ord_ in ('foo', -7, -1.1, 1.1):
            with self.assertRaisesRegex(ValueError, "'ord' must be a supported matrix norm"):
                linalg_ops.norm(matrix, ord=ord_, axis=[-2, -1])

    @test_util.run_v1_only('b/120545219')
    def testInvalidAxis(self):
        if False:
            while True:
                i = 10
        matrix = [[0.0, 1.0], [2.0, 3.0]]
        for axis_ in ([], [1, 2, 3], [[1]], [[1], [2]], [3.1415], [1, 1]):
            error_prefix = "'axis' must be None, an integer, or a tuple of 2 unique integers"
            with self.assertRaisesRegex(ValueError, error_prefix):
                linalg_ops.norm(matrix, axis=axis_)

def _GetNormOpTest(dtype_, shape_, ord_, axis_, keep_dims_, use_static_shape_):
    if False:
        while True:
            i = 10

    def _CompareNorm(self, matrix):
        if False:
            return 10
        np_norm = np.linalg.norm(matrix, ord=ord_, axis=axis_, keepdims=keep_dims_)
        with self.cached_session() as sess:
            if use_static_shape_:
                tf_matrix = constant_op.constant(matrix)
                tf_norm = linalg_ops.norm(tf_matrix, ord=ord_, axis=axis_, keepdims=keep_dims_)
                tf_norm_val = self.evaluate(tf_norm)
            else:
                tf_matrix = array_ops.placeholder(dtype_)
                tf_norm = linalg_ops.norm(tf_matrix, ord=ord_, axis=axis_, keepdims=keep_dims_)
                tf_norm_val = sess.run(tf_norm, feed_dict={tf_matrix: matrix})
        self.assertAllClose(np_norm, tf_norm_val, rtol=1e-05, atol=1e-05)

    @test_util.run_v1_only('b/120545219')
    def Test(self):
        if False:
            i = 10
            return i + 15
        is_matrix_norm = (isinstance(axis_, tuple) or isinstance(axis_, list)) and len(axis_) == 2
        is_fancy_p_norm = np.isreal(ord_) and np.floor(ord_) != ord_
        if not is_matrix_norm and ord_ == 'fro' or (is_matrix_norm and is_fancy_p_norm):
            self.skipTest('Not supported by neither numpy.linalg.norm nor tf.norm')
        if ord_ == 'euclidean' or (axis_ is None and len(shape) > 2):
            self.skipTest('Not supported by numpy.linalg.norm')
        matrix = np.random.randn(*shape_).astype(dtype_)
        if dtype_ in (np.complex64, np.complex128):
            matrix += 1j * np.random.randn(*shape_).astype(dtype_)
        _CompareNorm(self, matrix)
    return Test
if __name__ == '__main__':
    for use_static_shape in (False, True):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            for rows in (2, 5):
                for cols in (2, 5):
                    for batch in ([], [2], [2, 3]):
                        shape = batch + [rows, cols]
                        for ord in ('euclidean', 'fro', 0.5, 1, 2, np.inf):
                            for axis in [None, (-2, -1), (-1, -2), -len(shape), 0, len(shape) - 1]:
                                for keep_dims in (False, True):
                                    name = '%s_%s_ord_%s_axis_%s_%s_%s' % (dtype.__name__, '_'.join(map(str, shape)), ord, axis, keep_dims, use_static_shape)
                                    _AddTest(NormOpTest, 'Norm_' + name, _GetNormOpTest(dtype, shape, ord, axis, keep_dims, use_static_shape))
    test_lib.main()