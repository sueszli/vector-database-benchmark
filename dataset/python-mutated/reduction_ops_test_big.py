"""Functional tests for reduction ops."""
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class BaseReductionTest(test.TestCase):

    def _tf_reduce(self, x, reduction_axes, keepdims):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class BigReductionTest(BaseReductionTest):
    """Test reductions for sum and boolean all over a wide range of shapes."""

    def _tf_reduce_max(self, x, reduction_axes, keepdims):
        if False:
            while True:
                i = 10
        return math_ops.reduce_max(x, reduction_axes, keepdims)

    def _tf_reduce_all(self, x, reduction_axes, keepdims):
        if False:
            for i in range(10):
                print('nop')
        return math_ops.reduce_all(x, reduction_axes, keepdims)

    def _tf_reduce_mean(self, x, reduction_axes, keepdims):
        if False:
            i = 10
            return i + 15
        return math_ops.reduce_mean(x, reduction_axes, keepdims)

    def _tf_reduce_sum(self, x, reduction_axes, keepdims):
        if False:
            print('Hello World!')
        return math_ops.reduce_sum(x, reduction_axes, keepdims)

    @test_util.run_deprecated_v1
    def testFloat32Sum(self):
        if False:
            return 10
        arr_ = np.ones([4097, 4097], dtype=np.float32)
        for size_x in [1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025, 4096, 4097]:
            for size_y in [1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025, 4096, 4097]:
                arr = arr_[0:size_x, 0:size_y]
                col_sum = np.ones([size_y], dtype=np.float32) * size_x
                row_sum = np.ones([size_x], dtype=np.float32) * size_y
                full_sum = np.ones([], dtype=np.float32) * size_x * size_y
                with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                    arr_placeholder = array_ops.placeholder(dtype=np.float32, shape=(size_x, size_y))
                    tf_row_sum = self._tf_reduce_sum(arr_placeholder, 1, False)
                    tf_col_sum = self._tf_reduce_sum(arr_placeholder, 0, False)
                    tf_full_sum = self._tf_reduce_sum(arr_placeholder, [0, 1], False)
                    (tf_out_row, tf_out_col, tf_out_full) = sess.run([tf_row_sum, tf_col_sum, tf_full_sum], {arr_placeholder: arr})
                self.assertAllClose(col_sum, tf_out_col)
                self.assertAllClose(row_sum, tf_out_row)
                self.assertAllClose(full_sum, tf_out_full)
        arr_ = np.ones([130, 130, 130], dtype=np.float32)
        for size_x in range(1, 130, 13):
            for size_y in range(1, 130, 13):
                for size_z in range(1, 130, 13):
                    arr = arr_[0:size_x, 0:size_y, 0:size_z]
                    sum_y = np.ones([size_x, size_z], dtype=np.float32)
                    sum_xz = np.ones([size_y], dtype=np.float32)
                    with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                        arr_placeholder = array_ops.placeholder(dtype=np.float32, shape=(size_x, size_y, size_z))
                        tf_sum_xz = self._tf_reduce_mean(arr_placeholder, [0, 2], False)
                        tf_sum_y = self._tf_reduce_mean(arr_placeholder, 1, False)
                        (tf_out_sum_xz, tf_out_sum_y) = sess.run([tf_sum_xz, tf_sum_y], {arr_placeholder: arr})
                    self.assertAllClose(sum_y, tf_out_sum_y)
                    self.assertAllClose(sum_xz, tf_out_sum_xz)

    @test_util.run_deprecated_v1
    def testFloat32Max(self):
        if False:
            return 10
        arr_ = np.random.uniform(low=-3, high=-1, size=[4105, 4105]).astype(np.float32)
        for size_x in [1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025, 4096, 4097]:
            for size_y in [1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025, 4096, 4097]:
                arr = arr_[0:size_x, 0:size_y]
                col_max = np.max(arr, axis=0)
                row_max = np.max(arr, axis=1)
                full_max = np.max(col_max)
                with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                    arr_placeholder = array_ops.placeholder(dtype=np.float32, shape=(size_x, size_y))
                    tf_row_max = self._tf_reduce_max(arr_placeholder, 1, False)
                    tf_col_max = self._tf_reduce_max(arr_placeholder, 0, False)
                    tf_full_max = self._tf_reduce_max(arr_placeholder, [0, 1], False)
                    (tf_out_row, tf_out_col, tf_out_full) = sess.run([tf_row_max, tf_col_max, tf_full_max], {arr_placeholder: arr})
                self.assertAllClose(col_max, tf_out_col)
                self.assertAllClose(row_max, tf_out_row)
                self.assertAllClose(full_max, tf_out_full)
        arr_ = np.random.uniform(low=-3, high=-1, size=[130, 130, 130]).astype(np.float32)
        for size_x in range(1, 130, 13):
            for size_y in range(1, 130, 13):
                for size_z in range(1, 130, 13):
                    arr = arr_[0:size_x, 0:size_y, 0:size_z]
                    sum_y = np.max(arr, axis=1)
                    sum_xz = np.max(arr, axis=(0, 2))
                    with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                        arr_placeholder = array_ops.placeholder(dtype=np.float32, shape=(size_x, size_y, size_z))
                        tf_sum_xz = self._tf_reduce_max(arr_placeholder, [0, 2], False)
                        tf_sum_y = self._tf_reduce_max(arr_placeholder, 1, False)
                        (tf_out_sum_xz, tf_out_sum_y) = sess.run([tf_sum_xz, tf_sum_y], {arr_placeholder: arr})
                    self.assertAllClose(sum_y, tf_out_sum_y)
                    self.assertAllClose(sum_xz, tf_out_sum_xz)

    @test_util.run_deprecated_v1
    def testBooleanAll(self):
        if False:
            while True:
                i = 10
        arr_ = np.ones([4097, 4097], dtype=np.bool_)
        for size_x in [1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025, 4096, 4097]:
            for size_y in [1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025, 4096, 4097]:
                arr = arr_[0:size_x, 0:size_y]
                col_sum = np.ones([size_y], dtype=np.bool_)
                row_sum = np.ones([size_x], dtype=np.bool_)
                full_sum = np.ones([1], dtype=np.bool_).reshape([])
                with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                    arr_placeholder = array_ops.placeholder(dtype=np.bool_, shape=(size_x, size_y))
                    tf_row_sum = self._tf_reduce_all(arr_placeholder, 1, False)
                    tf_col_sum = self._tf_reduce_all(arr_placeholder, 0, False)
                    tf_full_sum = self._tf_reduce_all(arr_placeholder, [0, 1], False)
                    (tf_out_row, tf_out_col, tf_out_full) = sess.run([tf_row_sum, tf_col_sum, tf_full_sum], {arr_placeholder: arr})
                self.assertAllClose(col_sum, tf_out_col)
                self.assertAllClose(row_sum, tf_out_row)
                self.assertAllClose(full_sum, tf_out_full)
        arr_ = np.ones([130, 130, 130], dtype=np.bool_)
        for size_x in range(1, 130, 13):
            for size_y in range(1, 130, 13):
                for size_z in range(1, 130, 13):
                    arr = arr_[0:size_x, 0:size_y, 0:size_z]
                    sum_y = np.ones([size_x, size_z], dtype=np.bool_)
                    sum_xz = np.ones([size_y], dtype=np.bool_)
                    with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                        arr_placeholder = array_ops.placeholder(dtype=np.bool_, shape=(size_x, size_y, size_z))
                        tf_sum_xz = self._tf_reduce_all(arr_placeholder, [0, 2], False)
                        tf_sum_y = self._tf_reduce_all(arr_placeholder, 1, False)
                        (tf_out_sum_xz, tf_out_sum_y) = sess.run([tf_sum_xz, tf_sum_y], {arr_placeholder: arr})
                    self.assertAllClose(sum_y, tf_out_sum_y)
                    self.assertAllClose(sum_xz, tf_out_sum_xz)
if __name__ == '__main__':
    test.main()