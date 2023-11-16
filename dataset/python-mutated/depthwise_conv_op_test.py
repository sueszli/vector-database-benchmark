"""Functional tests for depthwise convolutional operations."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad
from tensorflow.python.platform import test

def ReferenceDepthwiseConv2D(input_tensor, filter_tensor, strides, padding, data_format=None):
    if False:
        i = 10
        return i + 15
    convs = []
    in_channels = filter_tensor.shape[2]
    for channel in range(in_channels):
        if data_format == 'NCHW':
            input_slice = input_tensor[:, channel:channel + 1, :, :]
        else:
            input_slice = input_tensor[:, :, :, channel:channel + 1]
        filter_slice = filter_tensor[:, :, channel:channel + 1, :]
        convs.append(nn_ops.conv2d(input_slice, filter_slice, strides, padding, data_format=data_format, name='depthwise_slice_%d' % channel))
    if data_format == 'NCHW':
        return array_ops.concat(convs, 1)
    else:
        return array_ops.concat(convs, 3)

def ConfigsToTest():
    if False:
        print('Hello World!')
    'Iterator for different convolution shapes, strides and paddings.\n\n  Yields:\n    Tuple (input_size, filter_size, out_size, stride, padding), the depthwise\n    convolution parameters.\n  '
    input_sizes = [[4, 5, 5, 48], [2, 5, 5, 48], [4, 8, 8, 84], [4, 17, 17, 48], [4, 9, 27, 8], [4, 31, 31, 7], [4, 35, 35, 2], [4, 147, 147, 2], [3, 299, 299, 3], [5, 183, 183, 1]]
    filter_sizes = [[1, 1, 48, 2], [2, 2, 48, 8], [1, 3, 84, 1], [3, 1, 48, 4], [3, 3, 8, 1], [3, 3, 7, 1], [5, 5, 2, 1], [3, 3, 2, 8], [2, 2, 3, 8], [5, 5, 1, 2]]
    out_sizes = [[4, 5, 5, 96], [2, 5, 5, 384], [4, 8, 8, 84], [4, 17, 17, 192], [4, 9, 27, 8], [4, 31, 31, 7], [4, 35, 35, 2], [4, 49, 49, 16], [3, 150, 150, 24], [5, 92, 92, 2]]
    strides = [1, 1, 1, 1, 1, 1, 1, 3, 2, 2]
    VALID = 'VALID'
    SAME = 'SAME'
    paddings = [SAME, SAME, SAME, SAME, SAME, SAME, SAME, VALID, SAME, SAME, SAME]
    for (i, f, o, s, p) in zip(input_sizes, filter_sizes, out_sizes, strides, paddings):
        yield (i, f, o, s, p)

def ConfigsWithDilationsToTest():
    if False:
        for i in range(10):
            print('nop')
    'Iterator for different convolution shapes, strides and paddings.\n\n  Yields:\n    Tuple (input_size, filter_size, out_size, stride, dilation, padding), the\n    depthwise\n    convolution parameters.\n  '
    input_sizes = [[4, 6, 6, 48], [4, 8, 8, 84], [4, 36, 36, 2], [4, 148, 148, 2], [3, 300, 300, 3]]
    filter_sizes = [[1, 1, 48, 2], [1, 3, 84, 1], [5, 5, 2, 1], [4, 4, 2, 8], [2, 2, 3, 8]]
    out_sizes = [[4, 6, 6, 96], [4, 8, 8, 84], [4, 36, 36, 2], [4, 74, 74, 16], [3, 296, 296, 24]]
    strides = [1, 1, 2, 2, 1]
    dilations = [2, 2, 4, 2, 4]
    VALID = 'VALID'
    SAME = 'SAME'
    paddings = [SAME, SAME, SAME, SAME, VALID]
    for (i, f, o, s, d, p) in zip(input_sizes, filter_sizes, out_sizes, strides, dilations, paddings):
        yield (i, f, o, s, d, p)

def CheckGradConfigsToTest():
    if False:
        return 10
    'Iterator for different convolution shapes, strides and paddings.\n\n  compute_gradient_error() is very expensive. So the configs should be\n  relatively small.\n\n  Yields:\n    Tuple (input_size, filter_size, out_size, stride, padding), the depthwise\n    convolution parameters.\n  '
    input_sizes = [[2, 5, 8, 1], [4, 5, 5, 1], [2, 4, 4, 2], [1, 15, 15, 2], [2, 15, 16, 1]]
    filter_sizes = [[4, 4, 1, 2], [2, 2, 1, 2], [3, 1, 2, 2], [1, 3, 2, 1], [3, 3, 1, 2]]
    out_sizes = [[2, 5, 8, 2], [4, 2, 2, 2], [2, 4, 4, 4], [1, 15, 15, 2], [2, 5, 5, 2]]
    strides = [1, 2, 1, 1, 3]
    VALID = 'VALID'
    SAME = 'SAME'
    paddings = [SAME, VALID, SAME, SAME, VALID]
    for (i, f, o, s, p) in zip(input_sizes, filter_sizes, out_sizes, strides, paddings):
        yield (i, f, o, s, p)

class DepthwiseConv2DTest(xla_test.XLATestCase):

    def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding, data_type, data_format='NHWC'):
        if False:
            i = 10
            return i + 15
        'Verifies the output values of the convolution function.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in\n        [batch, input_rows, input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in\n        [filter_rows, filter_cols, input_depth, depth_multiplier].\n      stride: Stride.\n      padding: Padding type.\n      data_type: The data type to use.\n      data_format: The data_format of the input. "NHWC" or "NCHW".\n    '
        total_size_1 = 1
        total_size_2 = 1
        for s in tensor_in_sizes:
            total_size_1 *= s
        for s in filter_in_sizes:
            total_size_2 *= s
        x1 = np.array([f * 1.0 for f in range(1, total_size_1 + 1)], dtype=data_type).reshape(tensor_in_sizes)
        x2 = np.array([f * 1.0 for f in range(1, total_size_2 + 1)], dtype=data_type).reshape(filter_in_sizes)
        with self.session() as sess:
            if data_type == np.float32:
                tolerance = 0.0001
            else:
                self.assertEqual(data_type, np.float64)
                tolerance = 1e-08
            t1 = array_ops.placeholder(shape=tensor_in_sizes, dtype=data_type)
            t2 = array_ops.placeholder(shape=filter_in_sizes, dtype=data_type)
            native_t1 = t1
            strides = [1, stride, stride, 1]
            if data_format == 'NCHW':
                native_t1 = array_ops.transpose(t1, [0, 3, 1, 2])
                strides = [1, 1, stride, stride]
            with self.test_scope():
                conv_native = nn_ops.depthwise_conv2d_native(native_t1, t2, strides=strides, data_format=data_format, padding=padding)
            if data_format == 'NCHW':
                conv_native = array_ops.transpose(conv_native, [0, 2, 3, 1])
            with ops.device('CPU'):
                conv_interface = ReferenceDepthwiseConv2D(t1, t2, strides=[1, stride, stride, 1], padding=padding)
            native_result = sess.run(conv_native, {t1: x1, t2: x2})
            interface_result = sess.run(conv_interface, {t1: x1, t2: x2})
        print('data_type:', data_type, 'max diff = ', np.amax(np.absolute(native_result - interface_result)))
        self.assertAllClose(np.ravel(native_result), np.ravel(interface_result), rtol=tolerance)

    @test_util.run_without_tensor_float_32('DepthwiseConv2D may use TF32 when available.')
    def testDepthwiseConv2D(self):
        if False:
            i = 10
            return i + 15
        for (index, (input_size, filter_size, _, stride, padding)) in enumerate(ConfigsToTest()):
            print('Testing DepthwiseConv2D,', index, 'th config:', input_size, '*', filter_size, 'stride:', stride, 'padding:', padding)
            for data_type in self.float_types:
                if data_type == np.float32:
                    self._VerifyValues(input_size, filter_size, stride, padding, data_type)

    @test_util.run_without_tensor_float_32('DepthwiseConv2D may use TF32 when available.')
    def testDepthwiseConv2DFormat(self):
        if False:
            print('Hello World!')
        for (index, (input_size, filter_size, _, stride, padding)) in enumerate(ConfigsToTest()):
            print('Testing DepthwiseConv2DFormat,', index, 'th config:', input_size, '*', filter_size, 'stride:', stride, 'padding:', padding)
            for data_type in self.float_types:
                if data_type == np.float32:
                    self._VerifyValues(input_size, filter_size, stride, padding, data_type, data_format='NCHW')

    def _VerifyHandValues(self, tensor_in_sizes, filter_in_sizes, stride, padding, expected):
        if False:
            return 10
        'Verifies the output values of the depthwise convolution function.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in\n        [batch, input_rows, input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in\n        [filter_rows, filter_cols, input_depth, depth_multiplier].\n      stride: Stride.\n      padding: Padding type.\n      expected: An array containing the expected operation outputs.\n    '
        total_size_1 = 1
        total_size_2 = 1
        for s in tensor_in_sizes:
            total_size_1 *= s
        for s in filter_in_sizes:
            total_size_2 *= s
        x1 = np.array([f * 1.0 for f in range(1, total_size_1 + 1)], dtype=np.float32).reshape(tensor_in_sizes)
        x2 = np.array([f * 1.0 for f in range(1, total_size_2 + 1)], dtype=np.float32).reshape(filter_in_sizes)
        with self.session() as sess:
            t1 = array_ops.placeholder(shape=tensor_in_sizes, dtype=np.float32)
            t2 = array_ops.placeholder(shape=filter_in_sizes, dtype=np.float32)
            with self.test_scope():
                conv = nn_ops.depthwise_conv2d_native(t1, t2, strides=[1, stride, stride, 1], padding=padding)
            value = sess.run(conv, {t1: x1, t2: x2})
        print('value = ', value)
        self.assertArrayNear(expected, np.ravel(value), 0.0001)
        self.assertShapeEqual(value, conv)

    def testConv2D2x2Filter(self):
        if False:
            while True:
                i = 10
        expected_output = [196, 216, 272, 296, 252, 280, 344, 376]
        self._VerifyHandValues(tensor_in_sizes=[1, 2, 3, 2], filter_in_sizes=[2, 2, 2, 2], stride=1, padding='VALID', expected=expected_output)

    def _VerifyValuesWithDilation(self, tensor_in_sizes, filter_in_sizes, stride, dilation, padding, data_type, data_format='NHWC'):
        if False:
            i = 10
            return i + 15
        'Verifies the output values of the convolution function.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,\n        input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in [filter_rows, filter_cols,\n        input_depth, depth_multiplier].\n      stride: Stride.\n      dilation: Dilation.\n      padding: Padding type.\n      data_type: The data type to use.\n      data_format: The data_format of the input. "NHWC" or "NCHW".\n    '
        total_size_1 = 1
        total_size_2 = 1
        for s in tensor_in_sizes:
            total_size_1 *= s
        for s in filter_in_sizes:
            total_size_2 *= s
        x1 = np.array([f * 1.0 for f in range(1, total_size_1 + 1)], dtype=data_type).reshape(tensor_in_sizes)
        x2 = np.array([f * 1.0 for f in range(1, total_size_2 + 1)], dtype=data_type).reshape(filter_in_sizes)
        with self.session() as sess:
            if data_type == np.float32:
                tolerance = 0.01
            else:
                self.assertEqual(data_type, np.float64)
                tolerance = 1e-08
            t1 = array_ops.placeholder(shape=tensor_in_sizes, dtype=data_type)
            t2 = array_ops.placeholder(shape=filter_in_sizes, dtype=data_type)
            native_t1 = t1
            strides = [1, stride, stride, 1]
            dilations = [dilation, dilation]
            if data_format == 'NCHW':
                native_t1 = array_ops.transpose(t1, [0, 3, 1, 2])
                strides = [1, 1, stride, stride]
            with self.test_scope():
                conv_native = nn_impl.depthwise_conv2d(native_t1, t2, strides=strides, rate=dilations, data_format=data_format, padding=padding)
            if data_format == 'NCHW':
                conv_native = array_ops.transpose(conv_native, [0, 2, 3, 1])
            with ops.device('CPU'):
                strides = [1, stride, stride, 1]
                conv_interface = nn_impl.depthwise_conv2d(t1, t2, strides=strides, rate=dilations, padding=padding)
            native_result = sess.run(conv_native, {t1: x1, t2: x2})
            interface_result = sess.run(conv_interface, {t1: x1, t2: x2})
        print('data_type:', data_type, 'max diff = ', np.amax(np.absolute(native_result - interface_result)))
        self.assertAllClose(np.ravel(native_result), np.ravel(interface_result), rtol=tolerance)

    def testDilationDepthwiseConv2DWith(self):
        if False:
            i = 10
            return i + 15
        for (index, (input_size, filter_size, _, stride, dilation, padding)) in enumerate(ConfigsWithDilationsToTest()):
            print('Testing DilationDepthwiseConv2D,', index, 'th config:', input_size, '*', filter_size, 'stride:', stride, 'dilation: ', dilation, 'padding:', padding)
            for data_type in self.float_types:
                if data_type == np.float32:
                    self._VerifyValuesWithDilation(input_size, filter_size, stride, dilation, padding, data_type)

    def testDilationDepthwiseConv2DWithFormat(self):
        if False:
            for i in range(10):
                print('nop')
        for (index, (input_size, filter_size, _, stride, dilation, padding)) in enumerate(ConfigsWithDilationsToTest()):
            print('Testing DilationDepthwiseConv2DFormat,', index, 'th config:', input_size, '*', filter_size, 'stride:', stride, 'dilation:', dilation, 'padding:', padding)
            for data_type in self.float_types:
                if data_type == np.float32:
                    self._VerifyValuesWithDilation(input_size, filter_size, stride, dilation, padding, data_type, data_format='NCHW')

    def _CompareBackpropInput(self, input_sizes, filter_sizes, output_sizes, stride, padding):
        if False:
            return 10
        x1 = np.random.rand(*filter_sizes).astype(np.float32)
        x2 = np.random.rand(*output_sizes).astype(np.float32)

        def _GetVal(use_xla):
            if False:
                while True:
                    i = 10
            with self.session():
                t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
                t1 = array_ops.placeholder(np.float32, shape=filter_sizes)
                t2 = array_ops.placeholder(np.float32, shape=output_sizes)
                if use_xla:
                    with self.test_scope():
                        backprop = nn_ops.depthwise_conv2d_native_backprop_input(t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
                else:
                    backprop = nn_ops.depthwise_conv2d_native_backprop_input(t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
                ret = backprop.eval({t1: x1, t2: x2})
                self.assertShapeEqual(ret, backprop)
                return ret
        gpu_value = _GetVal(use_xla=True)
        cpu_value = _GetVal(use_xla=False)
        self.assertAllClose(cpu_value, gpu_value, rtol=0.001, atol=0.001)

    def testDepthwiseConv2DInputGradCompare(self):
        if False:
            while True:
                i = 10
        for (index, (input_size, filter_size, output_size, stride, padding)) in enumerate(ConfigsToTest()):
            print('Testing DepthwiseConv2DInputGradCompare,', index, 'th config:', input_size, '*', filter_size, 'stride:', stride, 'padding:', padding)
            self._CompareBackpropInput(input_size, filter_size, output_size, stride, padding)

    def _CompareBackpropFilter(self, input_sizes, filter_sizes, output_sizes, stride, padding, data_format='NHWC'):
        if False:
            while True:
                i = 10
        x0 = np.random.rand(*input_sizes).astype(np.float32)
        x2 = np.random.rand(*output_sizes).astype(np.float32)

        def _GetVal(use_xla):
            if False:
                i = 10
                return i + 15
            with self.session():
                t0 = array_ops.placeholder(np.float32, shape=input_sizes)
                t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
                t2 = array_ops.placeholder(np.float32, shape=output_sizes)
                native_t0 = t0
                native_t2 = t2
                strides = [1, stride, stride, 1]
                if use_xla:
                    if data_format == 'NCHW':
                        native_t0 = array_ops.transpose(t0, [0, 3, 1, 2])
                        native_t2 = array_ops.transpose(t2, [0, 3, 1, 2])
                        strides = [1, 1, stride, stride]
                    with self.test_scope():
                        backprop = nn_ops.depthwise_conv2d_native_backprop_filter(native_t0, t1, native_t2, strides=strides, padding=padding, data_format=data_format)
                else:
                    backprop = nn_ops.depthwise_conv2d_native_backprop_filter(native_t0, t1, native_t2, strides=strides, padding=padding)
                ret = backprop.eval({t0: x0, t2: x2})
                self.assertShapeEqual(ret, backprop)
                return ret
        gpu_value = _GetVal(use_xla=True)
        cpu_value = _GetVal(use_xla=False)
        self.assertAllClose(cpu_value, gpu_value, rtol=0.0001, atol=0.0001)

    @test_util.run_without_tensor_float_32('DepthwiseConv2DFilterGrad may use TF32 when available.')
    def testDepthwiseConv2DFilterGradCompare(self):
        if False:
            while True:
                i = 10
        for (index, (input_size, filter_size, output_size, stride, padding)) in enumerate(ConfigsToTest()):
            print('Testing DepthwiseConv2DFilterGradCompare,', index, 'th config:', input_size, '*', filter_size, 'producing output', output_size, 'stride:', stride, 'padding:', padding)
            self._CompareBackpropFilter(input_size, filter_size, output_size, stride, padding)

    @test_util.run_without_tensor_float_32('DepthwiseConv2DFilterGrad may use TF32 when available.')
    def testDepthwiseConv2DFilterGradFormatNCHWCompare(self):
        if False:
            return 10
        for (index, (input_size, filter_size, output_size, stride, padding)) in enumerate(ConfigsToTest()):
            print('Testing DepthwiseConv2DFilterGradFormatNCHWCompare,', index, 'th config:', input_size, '*', filter_size, 'producing output', output_size, 'stride:', stride, 'padding:', padding)
            self._CompareBackpropFilter(input_size, filter_size, output_size, stride, padding, data_format='NCHW')

    def _CompareBackpropInputWithDilation(self, input_sizes, filter_sizes, output_sizes, stride, dilation, padding):
        if False:
            return 10
        x1 = np.random.rand(*filter_sizes).astype(np.float32)
        x2 = np.random.rand(*output_sizes).astype(np.float32)

        def _GetVal(use_xla):
            if False:
                i = 10
                return i + 15
            with self.session():
                t1 = array_ops.placeholder(np.float32, shape=filter_sizes)
                t2 = array_ops.placeholder(np.float32, shape=output_sizes)
                if use_xla:
                    with self.test_scope():
                        t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
                        backprop = nn_ops.depthwise_conv2d_native_backprop_input(t0, t1, t2, strides=[1, stride, stride, 1], dilations=[1, dilation, dilation, 1], padding=padding)
                else:
                    t3 = array_ops.space_to_batch(t2, block_size=dilation, paddings=[[0, 0], [0, 0]])
                    input_sizes_transform = [input_sizes[0] * dilation * dilation, input_sizes[1] // dilation, input_sizes[2] // dilation, input_sizes[3]]
                    t0 = constant_op.constant(input_sizes_transform, shape=[len(input_sizes)])
                    backprop_naive = nn_ops.depthwise_conv2d_native_backprop_input(t0, t1, t3, strides=[1, stride, stride, 1], padding=padding)
                    backprop = array_ops.batch_to_space(backprop_naive, [[0, 0], [0, 0]], block_size=dilation)
                ret = backprop.eval({t1: x1, t2: x2})
                self.assertShapeEqual(ret, backprop)
                return ret
        gpu_value = _GetVal(use_xla=True)
        cpu_value = _GetVal(use_xla=False)
        self.assertAllClose(cpu_value, gpu_value, rtol=0.01, atol=0.001)

    def testDilationDepthwiseConv2DInputGradWithCompare(self):
        if False:
            for i in range(10):
                print('nop')
        for (index, (input_size, filter_size, output_size, stride, dilation, padding)) in enumerate(ConfigsWithDilationsToTest()):
            print('Testing DilationDepthwiseConv2DInputGradWithDilationCompare,', index, 'th config:', input_size, '*', filter_size, 'stride:', stride, 'dilation:', dilation, 'padding:', padding)
            if stride == 1:
                self._CompareBackpropInputWithDilation(input_size, filter_size, output_size, stride, dilation, padding)

    def _CompareBackpropFilterWithDilation(self, input_sizes, filter_sizes, output_sizes, stride, dilation, padding, data_format='NHWC'):
        if False:
            for i in range(10):
                print('nop')
        x0 = np.random.rand(*input_sizes).astype(np.float32)
        x2 = np.random.rand(*output_sizes).astype(np.float32)

        def _GetVal(use_xla):
            if False:
                while True:
                    i = 10
            with self.session():
                t0 = array_ops.placeholder(np.float32, shape=input_sizes)
                t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
                t2 = array_ops.placeholder(np.float32, shape=output_sizes)
                native_t0 = t0
                native_t2 = t2
                strides = [1, stride, stride, 1]
                dilations = [1, dilation, dilation, 1]
                if use_xla:
                    if data_format == 'NCHW':
                        native_t0 = array_ops.transpose(t0, [0, 3, 1, 2])
                        native_t2 = array_ops.transpose(t2, [0, 3, 1, 2])
                        strides = [1, 1, stride, stride]
                        dilations = [1, 1, dilation, dilation]
                    with self.test_scope():
                        backprop = nn_ops.depthwise_conv2d_native_backprop_filter(native_t0, t1, native_t2, strides=strides, padding=padding, dilations=dilations, data_format=data_format)
                else:
                    native_t3 = array_ops.space_to_batch(native_t2, block_size=dilation, paddings=[[0, 0], [0, 0]])
                    native_t0_transform = array_ops.space_to_batch(native_t0, block_size=dilation, paddings=[[0, 0], [0, 0]])
                    backprop = nn_ops.depthwise_conv2d_native_backprop_filter(native_t0_transform, t1, native_t3, strides=strides, padding=padding)
                ret = backprop.eval({t0: x0, t2: x2})
                self.assertShapeEqual(ret, backprop)
                return ret
        gpu_value = _GetVal(use_xla=True)
        cpu_value = _GetVal(use_xla=False)
        self.assertAllClose(cpu_value, gpu_value, rtol=0.001, atol=0.0001)

    def testDilationDepthwiseConv2DFilterGradCompare(self):
        if False:
            return 10
        for (index, (input_size, filter_size, output_size, stride, dilation, padding)) in enumerate(ConfigsWithDilationsToTest()):
            print('Testing DilationDepthwiseConv2DFilterGradCompare,', index, 'th config:', input_size, '*', filter_size, 'producing output', output_size, 'stride:', stride, 'dilation:', dilation, 'padding:', padding)
            if stride == 1:
                self._CompareBackpropFilterWithDilation(input_size, filter_size, output_size, stride, dilation, padding)
if __name__ == '__main__':
    test.main()