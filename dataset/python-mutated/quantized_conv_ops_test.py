"""Functional tests for quantized convolutional operations."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

class Conv2DTest(test.TestCase):

    def __init__(self, method_name='runTest'):
        if False:
            while True:
                i = 10
        super(Conv2DTest, self).__init__(method_name)

    def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding, expected):
        if False:
            return 10
        'Verifies the output values of the convolution function.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in\n        [batch, input_rows, input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in\n        [kernel_rows, kernel_cols, input_depth, output_depth].\n      stride: Stride.\n      padding: Padding type.\n      expected: An array containing the expected operation outputs.\n    '
        total_size_1 = 1
        total_size_2 = 1
        for s in tensor_in_sizes:
            total_size_1 *= s
        for s in filter_in_sizes:
            total_size_2 *= s
        x1 = np.array([f for f in range(1, total_size_1 + 1)])
        x1 = x1.astype(np.uint8).reshape(tensor_in_sizes)
        x1_min = 0.0
        x1_max = 255.0
        x2 = np.array([f for f in range(1, total_size_2 + 1)]).astype(np.uint8)
        x2 = x2.astype(np.uint8).reshape(filter_in_sizes)
        x2_min = 0.0
        x2_max = 255.0
        with self.cached_session(use_gpu=False) as sess:
            t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtypes.quint8)
            t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtypes.quint8)
            conv = nn_ops.quantized_conv2d(t1, t2, out_type=dtypes.qint32, strides=[1, stride, stride, 1], padding=padding, min_input=x1_min, max_input=x1_max, min_filter=x2_min, max_filter=x2_max)
            value = self.evaluate(conv)
        quantized_output = value[0]
        output_min = value[1]
        output_max = value[2]
        float_output = self._QuantizedOutputToFloat(quantized_output, output_min, output_max)
        self.assertArrayNear(expected, float_output.flatten(), 1.0)
        self.assertEqual(value[0].shape, conv[0].get_shape())

    def _assertQuantizedArrayEquals(self, iarray1, iarray2):
        if False:
            i = 10
            return i + 15
        for (i1, i2) in zip(iarray1, iarray2):
            self.assertTrue(i1 == i2)

    def _QuantizedOutputToFloat(self, quantized, quantized_min, quantized_max):
        if False:
            return 10
        number_of_bits = 32
        number_of_steps = 1 << number_of_bits
        range_adjust = number_of_steps / (number_of_steps - 1.0)
        quantized_range = (quantized_max - quantized_min) * range_adjust
        range_scale = quantized_range / number_of_steps
        lowest_quantized = -(1 << number_of_bits - 1)
        result = np.array([quantized_min + (float(x) - lowest_quantized) * range_scale for x in quantized.flatten()])
        return result

    def testConv2D1x1Filter(self):
        if False:
            return 10
        expected_output = [30, 36, 42, 66, 81, 96, 102, 126, 150, 138, 171, 204, 174, 216, 258, 210, 261, 312]
        self._VerifyValues(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[1, 1, 3, 3], stride=1, padding='VALID', expected=expected_output)

    def testConv2D2x2Filter(self):
        if False:
            print('Hello World!')
        expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
        self._VerifyValues(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 3], stride=1, padding='VALID', expected=expected_output)

    def testConv2D1x2Filter(self):
        if False:
            for i in range(10):
                print('nop')
        expected_output = [231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0, 936.0, 1029.0]
        self._VerifyValues(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[1, 2, 3, 3], stride=1, padding='VALID', expected=expected_output)

    def testConv2D2x2FilterStride2(self):
        if False:
            print('Hello World!')
        expected_output = [2271.0, 2367.0, 2463.0]
        self._VerifyValues(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 3], stride=2, padding='VALID', expected=expected_output)

    def testConv2D2x2FilterStride2Same(self):
        if False:
            return 10
        expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
        self._VerifyValues(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 3], stride=2, padding='SAME', expected=expected_output)

    def _testBadInputSize(self, tin=None, tfilter=None, min_input=None, max_input=None, min_filter=None, max_filter=None, error_regex=''):
        if False:
            while True:
                i = 10
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        if tin is None:
            tin = math_ops.cast(constant_op.constant(1, shape=[1, 2, 3, 3]), dtype=dtypes.quint8)
        if tfilter is None:
            tfilter = math_ops.cast(constant_op.constant(1, shape=[1, 2, 3, 3]), dtype=dtypes.quint8)
        if min_input is None:
            min_input = constant_op.constant(0, shape=[], dtype=dtypes.float32)
        if max_input is None:
            max_input = constant_op.constant(0, shape=[], dtype=dtypes.float32)
        if min_filter is None:
            min_filter = constant_op.constant(0, shape=[], dtype=dtypes.float32)
        if max_filter is None:
            max_filter = constant_op.constant(0, shape=[], dtype=dtypes.float32)
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), error_regex):
            self.evaluate(nn_ops.quantized_conv2d(tin, tfilter, out_type=dtypes.qint32, strides=strides, padding=padding, min_input=min_input, max_input=max_input, min_filter=min_filter, max_filter=max_filter))

    def testBadInputSizes(self):
        if False:
            return 10
        self._testBadInputSize(tin=math_ops.cast(constant_op.constant(1, shape=[1, 2]), dtype=dtypes.quint8), error_regex='must be rank 4')
        self._testBadInputSize(tfilter=math_ops.cast(constant_op.constant(1, shape=[1, 2]), dtype=dtypes.quint8), error_regex='must be rank 4')
        self._testBadInputSize(min_input=constant_op.constant(0, shape=[1], dtype=dtypes.float32), error_regex='must be rank 0')
        self._testBadInputSize(max_input=constant_op.constant(0, shape=[1], dtype=dtypes.float32), error_regex='must be rank 0')
        self._testBadInputSize(min_filter=constant_op.constant(0, shape=[1], dtype=dtypes.float32), error_regex='must be rank 0')
        self._testBadInputSize(max_filter=constant_op.constant(0, shape=[1], dtype=dtypes.float32), error_regex='must be rank 0')
if __name__ == '__main__':
    test.main()