"""Tests for convolution related functionality in tensorflow.ops.nn."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad
from tensorflow.python.platform import test

class Conv1DTransposeTest(test.TestCase):

    def testConv1DTransposeSingleStride(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            strides = [1, 1, 1]
            x_shape = [2, 6, 3]
            y_shape = [2, 6, 2]
            f_shape = [3, 2, 3]
            x = constant_op.constant(1.0, shape=x_shape, name='x', dtype=dtypes.float32)
            f = constant_op.constant(1.0, shape=f_shape, name='filter', dtype=dtypes.float32)
            output = nn_ops.conv1d_transpose(x, f, y_shape, strides=strides, padding='SAME')
            value = self.evaluate(output)
            for n in range(y_shape[0]):
                for w in range(y_shape[1]):
                    for c in range(y_shape[2]):
                        target = 2 * 3.0
                        w_in = w > 0 and w < y_shape[1] - 1
                        if w_in:
                            target += 3.0
                        self.assertAllClose(target, value[n, w, c])

    def testConv1DTransposeSame(self):
        if False:
            return 10
        with self.cached_session():
            strides = [1, 2, 1]
            x_shape = [2, 4, 3]
            y_shape = [2, 8, 2]
            f_shape = [3, 2, 3]
            x = constant_op.constant(1.0, shape=x_shape, name='x', dtype=dtypes.float32)
            f = constant_op.constant(1.0, shape=f_shape, name='filter', dtype=dtypes.float32)
            output = nn_ops.conv1d_transpose(x, f, y_shape, strides=strides, padding='SAME')
            value = self.evaluate(output)
            for n in range(x_shape[0]):
                for k in range(f_shape[1]):
                    for w in range(y_shape[1]):
                        target = 3.0
                        w_in = w % strides[1] == 0 and w > 0 and (w < y_shape[1] - 1)
                        if w_in:
                            target += 3.0
                        self.assertAllClose(target, value[n, w, k])

    def testConv1DTransposeValid(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            strides = [1, 2, 1]
            x_shape = [2, 4, 3]
            y_shape = [2, 9, 2]
            f_shape = [3, 2, 3]
            x = constant_op.constant(1.0, shape=x_shape, name='x', dtype=dtypes.float32)
            f = constant_op.constant(1.0, shape=f_shape, name='filter', dtype=dtypes.float32)
            output = nn_ops.conv1d_transpose(x, f, y_shape, strides=strides, padding='VALID')
            value = self.evaluate(output)
            cache_values = np.zeros(y_shape, dtype=np.float32)
            pad = 1
            for n in range(x_shape[0]):
                for k in range(f_shape[1]):
                    for w in range(pad, y_shape[1] - pad):
                        target = 3.0
                        w_in = w % strides[1] == 0 and w > pad and (w < y_shape[1] - 1 - pad)
                        if w_in:
                            target += 3.0
                        cache_values[n, w, k] = target
                    cache_values[n, 0, k] = cache_values[n, 1, k]
                    cache_values[n, -1, k] = cache_values[n, -2, k]
                    cache_values[n, :, k] = cache_values[n, :, k]
        self.assertAllClose(cache_values, value)

    @test_util.run_deprecated_v1
    def testGradient(self):
        if False:
            while True:
                i = 10
        self.skipTest('b/262851489: Fix nightly build for GPU.')
        x_shape = [2, 4, 3]
        f_shape = [3, 2, 3]
        y_shape = [2, 8, 2]
        strides = [1, 2, 1]
        np.random.seed(1)
        x_val = np.random.random_sample(x_shape).astype(np.float64)
        f_val = np.random.random_sample(f_shape).astype(np.float64)
        with self.cached_session():
            x = constant_op.constant(x_val, name='x', dtype=dtypes.float32)
            f = constant_op.constant(f_val, name='f', dtype=dtypes.float32)
            output = nn_ops.conv1d_transpose(x, f, y_shape, strides=strides, padding='SAME')
            err = gradient_checker.compute_gradient_error([x, f], [x_shape, f_shape], output, y_shape)
        print('conv1d_transpose gradient err = %g ' % err)
        err_tolerance = 0.0005
        self.assertLess(err, err_tolerance)

    def testConv1DTransposeSingleStrideNCW(self):
        if False:
            return 10
        if test.is_gpu_available(cuda_only=True):
            with self.session():
                strides = [1, 1, 1]
                x_shape = [2, 3, 4]
                y_shape = [2, 2, 4]
                f_shape = [3, 2, 3]
                x = constant_op.constant(1.0, shape=x_shape, name='x', dtype=dtypes.float32)
                f = constant_op.constant(1.0, shape=f_shape, name='filter', dtype=dtypes.float32)
                output = nn_ops.conv1d_transpose(x, f, y_shape, strides=strides, padding='SAME', data_format='NCW')
                value = self.evaluate(output)
                for n in range(x_shape[0]):
                    for k in range(f_shape[1]):
                        for w in range(y_shape[2]):
                            target = 2 * 3.0
                            w_in = w > 0 and w < y_shape[2] - 1
                            if w_in:
                                target += 3.0
                            self.assertAllClose(target, value[n, k, w])

    def testConv1DTransposeSameNCW(self):
        if False:
            i = 10
            return i + 15
        if test.is_gpu_available(cuda_only=True):
            with self.session():
                strides = [1, 1, 2]
                x_shape = [2, 3, 4]
                y_shape = [2, 2, 8]
                f_shape = [3, 2, 3]
                x = constant_op.constant(1.0, shape=x_shape, name='x', dtype=dtypes.float32)
                f = constant_op.constant(1.0, shape=f_shape, name='filter', dtype=dtypes.float32)
                output = nn_ops.conv1d_transpose(x, f, y_shape, strides=strides, padding='SAME', data_format='NCW')
                value = self.evaluate(output)
                for n in range(x_shape[0]):
                    for k in range(f_shape[1]):
                        for w in range(y_shape[2]):
                            target = 3.0
                            w_in = w % strides[2] == 0 and w > 0 and (w < y_shape[2] - 1)
                            if w_in:
                                target += 3.0
                            self.assertAllClose(target, value[n, k, w])

    def testConv1DTransposeValidNCW(self):
        if False:
            return 10
        if test.is_gpu_available(cuda_only=True):
            with self.session():
                strides = [1, 1, 2]
                x_shape = [2, 3, 4]
                y_shape = [2, 2, 9]
                f_shape = [3, 2, 3]
                x = constant_op.constant(1.0, shape=x_shape, name='x', dtype=dtypes.float32)
                f = constant_op.constant(1.0, shape=f_shape, name='filter', dtype=dtypes.float32)
                output = nn_ops.conv1d_transpose(x, f, y_shape, strides=strides, padding='VALID', data_format='NCW')
                value = self.evaluate(output)
                cache_values = np.zeros(y_shape, dtype=np.float32)
                pad = 1
                for n in range(x_shape[0]):
                    for k in range(f_shape[1]):
                        for w in range(pad, y_shape[2] - pad):
                            target = 3.0
                            w_in = w % strides[2] == 0 and w > pad and (w < y_shape[2] - 1 - pad)
                            if w_in:
                                target += 3.0
                            cache_values[n, k, w] = target
                        cache_values[n, k, 0] = cache_values[n, k, 1]
                        cache_values[n, k, -1] = cache_values[n, k, -2]
                        cache_values[n, k, :] = cache_values[n, k, :]
                self.assertAllClose(cache_values, value)
if __name__ == '__main__':
    test.main()