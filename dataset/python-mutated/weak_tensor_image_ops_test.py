"""Tests for tensorflow.ops.image_ops on WeakTensor."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import weak_tensor_ops
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.platform import test
_get_weak_tensor = weak_tensor_test_util.get_weak_tensor

class AdjustBrightnessTest(test.TestCase):

    def _testBrightness(self, x_np, y_np, delta, tol=1e-06):
        if False:
            print('Hello World!')
        with self.cached_session():
            x = _get_weak_tensor(x_np, shape=x_np.shape)
            y = image_ops.adjust_brightness(x, delta)
            y_tf = self.evaluate(y)
            self.assertIsInstance(y, WeakTensor)
            self.assertAllClose(y_tf, y_np, tol)

    def testPositiveDeltaFloat32(self):
        if False:
            print('Hello World!')
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.float32).reshape(x_shape) / 255.0
        y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
        y_np = np.array(y_data, dtype=np.float32).reshape(x_shape) / 255.0
        self._testBrightness(x_np, y_np, delta=10.0 / 255.0)

    def testPositiveDeltaFloat64(self):
        if False:
            while True:
                i = 10
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.float64).reshape(x_shape) / 255.0
        y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
        y_np = np.array(y_data, dtype=np.float64).reshape(x_shape) / 255.0
        self._testBrightness(x_np, y_np, delta=10.0 / 255.0, tol=0.001)

class AdjustGamma(test.TestCase):

    def test_adjust_gamma_less_zero_float32(self):
        if False:
            while True:
                i = 10
        'White image should be returned for gamma equal to zero.'
        with self.cached_session():
            x_data = np.random.uniform(0, 1.0, (8, 8))
            x_np = np.array(x_data, dtype=np.float32)
            x = _get_weak_tensor(x_np, shape=x_np.shape)
            err_msg = 'Gamma should be a non-negative real number'
            with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), err_msg):
                image_ops.adjust_gamma(x, gamma=-1)

    def test_adjust_gamma_less_zero_tensor(self):
        if False:
            i = 10
            return i + 15
        'White image should be returned for gamma equal to zero.'
        with self.cached_session():
            x_data = np.random.uniform(0, 1.0, (8, 8))
            x_np = np.array(x_data, dtype=np.float32)
            x = _get_weak_tensor(x_np, shape=x_np.shape)
            y = constant_op.constant(-1.0, dtype=dtypes.float32)
            err_msg = 'Gamma should be a non-negative real number'
            with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), err_msg):
                image = image_ops.adjust_gamma(x, gamma=y)
                self.evaluate(image)

    def _test_adjust_gamma_float32(self, gamma):
        if False:
            print('Hello World!')
        'Verifying the output with expected results for gamma correction for float32 images.'
        with self.cached_session():
            x_np = np.random.uniform(0, 1.0, (8, 8))
            x = _get_weak_tensor(x_np, shape=x_np.shape)
            y = image_ops.adjust_gamma(x, gamma=gamma)
            y_tf = self.evaluate(y)
            self.assertIsInstance(y, WeakTensor)
            y_np = np.clip(np.power(x_np, gamma), 0, 1.0)
            self.assertAllClose(y_tf, y_np, 1e-06)

    def test_adjust_gamma_one_float32(self):
        if False:
            return 10
        'Same image should be returned for gamma equal to one.'
        self._test_adjust_gamma_float32(1.0)

    def test_adjust_gamma_less_one_float32(self):
        if False:
            for i in range(10):
                print('nop')
        'Verifying the output with expected results for gamma correction with gamma equal to half for float32 images.'
        self._test_adjust_gamma_float32(0.5)

    def test_adjust_gamma_greater_one_float32(self):
        if False:
            i = 10
            return i + 15
        'Verifying the output with expected results for gamma correction with gamma equal to two for float32 images.'
        self._test_adjust_gamma_float32(1.0)

    def test_adjust_gamma_zero_float32(self):
        if False:
            print('Hello World!')
        'White image should be returned for gamma equal to zero for float32 images.'
        self._test_adjust_gamma_float32(0.0)
if __name__ == '__main__':
    ops.set_dtype_conversion_mode('all')
    test.main()