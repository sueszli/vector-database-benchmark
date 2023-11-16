"""Tests for Dequantize Operations."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class DequantizeOpTest(test.TestCase):

    def __init__(self, method_name='runTest'):
        if False:
            while True:
                i = 10
        super(DequantizeOpTest, self).__init__(method_name)

    def _testDequantizeOp(self, inputs, min_range, max_range, dtype, mode='MIN_COMBINED', narrow_range=False):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            input_op = constant_op.constant(inputs, shape=[len(inputs)], dtype=dtype)
            dequantized = array_ops.dequantize(input_op, min_range, max_range, mode=mode, narrow_range=narrow_range)
            tf_ans = self.evaluate(dequantized)
        type_dict = {dtypes.quint8: np.uint8, dtypes.qint8: np.int8, dtypes.quint16: np.uint16, dtypes.qint16: np.int16}
        self.assertIn(dtype, type_dict.keys())
        v_max = np.iinfo(type_dict[dtype]).max
        v_min = np.iinfo(type_dict[dtype]).min
        self.assertGreaterEqual(min_range, v_min)
        self.assertLessEqual(max_range, v_max)
        type_range = v_max - v_min
        if mode == 'MIN_COMBINED':
            if v_min < 0:
                half_range = (type_range + 1) / 2
            else:
                half_range = 0.0
            np_ans = (inputs.astype(np.float32) + half_range) * (max_range - min_range) / type_range + min_range
        elif mode == 'SCALED':
            if narrow_range:
                v_min += 1
            scale_factor = max(min_range / v_min, max_range / v_max)
            np_ans = inputs.astype(np.float32) * scale_factor
        self.assertAllClose(tf_ans, np_ans, rtol=1e-05, atol=1e-05)

    def testBasicQuint8(self):
        if False:
            return 10
        self._testDequantizeOp(np.array([0, 128, 255]), 0.0, 6.0, dtypes.quint8)
        self._testDequantizeOp(np.array([0, 128, 255]), 0.0, 123.456, dtypes.quint8)
        self._testDequantizeOp(np.array([0, 4, 42, 108, 243]), 5.0, 200.2, dtypes.quint8)

    def testBasicQint8(self):
        if False:
            i = 10
            return i + 15
        self._testDequantizeOp(np.array([-128, 0, 127]), -1.0, 2.0, dtypes.qint8)
        self._testDequantizeOp(np.array([-2, 4, -17]), -5.0, -3.0, dtypes.qint8)
        self._testDequantizeOp(np.array([0, -4, 42, -108]), 5.0, 40.0, dtypes.qint8)

    def testScaledMode(self):
        if False:
            print('Hello World!')
        self._testDequantizeOp(np.array([-128, 0, 127]), -1.0, 2.0, dtypes.qint8, mode='SCALED')
        self._testDequantizeOp(np.array([-2, 4, -17]), -5.0, -3.0, dtypes.qint8, mode='SCALED')
        self._testDequantizeOp(np.array([0, -4, 42, -108]), 5.0, 40.0, dtypes.qint8, mode='SCALED')

    def testNarrowRange(self):
        if False:
            print('Hello World!')
        self._testDequantizeOp(np.array([-128, 0, 127]), -1.0, 2.0, dtypes.qint8, mode='SCALED', narrow_range=True)
        self._testDequantizeOp(np.array([-2, 4, -17]), -5.0, -3.0, dtypes.qint8, mode='SCALED', narrow_range=True)
        self._testDequantizeOp(np.array([0, -4, 42, -108]), 5.0, 40.0, dtypes.qint8, mode='SCALED', narrow_range=True)

    def testAxis(self):
        if False:
            while True:
                i = 10

        def scale_per_slice(shape, axis, values):
            if False:
                return 10
            out = np.take(values, np.remainder(np.arange(np.prod(shape)), len(values))).reshape(shape)
            if axis is not None:
                scale_shape = [1] * len(shape)
                scale_shape[axis] = shape[axis]
                out *= np.arange(1, shape[axis] + 1).reshape(scale_shape)
            return out
        shape = np.array([2, 3, 4, 5])
        values = np.array([-128, -64, 0, 38, 102, 71, 64], dtype=np.int32)
        dequant_values = np.array([-2, -1.0, 0, 0.59375, 1.59375, 1.109375, 1.0], dtype=np.float32)
        for axis in [None, 0, 1, 2, 3]:
            inputs = constant_op.constant(scale_per_slice(shape, None, values), dtype=dtypes.qint8)
            expected_dequantized = scale_per_slice(shape, axis, dequant_values)
            if axis is None:
                (min_range, max_range) = (-2.0, 1.6)
            else:
                num_slices = shape[axis]
                (min_range, max_range) = ([], [])
                for slice_idx in range(num_slices):
                    min_range.append(-2.0 * (slice_idx + 1))
                    max_range.append(1.6 * (slice_idx + 1))
            dequantized = self.evaluate(array_ops.dequantize(inputs, min_range, max_range, mode='SCALED', axis=axis))
            self.assertAllEqual(dequantized, expected_dequantized)
            if axis is not None:
                dequantized = self.evaluate(array_ops.dequantize(inputs, min_range, max_range, mode='SCALED', axis=axis - 4))
                self.assertAllClose(dequantized, expected_dequantized)
if __name__ == '__main__':
    test.main()