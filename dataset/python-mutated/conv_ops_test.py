"""Functional tests for convolutional operations."""
import os
import time
from absl.testing import parameterized
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.compat import collections_abc

def GetShrunkInceptionShapes(shrink=10):
    if False:
        return 10
    'Iterator for smaller versions of convolution shapes in 2015 Inception.\n\n  Relative to inception, each depth value is `depth // shrink`.\n\n  Args:\n    shrink: Factor to shrink each depth value by relative to Inception.\n\n  Yields:\n    Tuple (input_size, filter_size, out_size, stride, padding), the convolution\n    parameters of Inception layers.\n  '
    input_sizes = [[4, 5, 5, 1248], [4, 8, 8, 384], [4, 8, 8, 384], [4, 8, 8, 2048], [4, 8, 8, 448], [4, 8, 8, 2048], [4, 8, 8, 2048], [4, 8, 8, 2048], [4, 8, 8, 1760], [4, 8, 8, 1760], [4, 8, 8, 1760], [4, 8, 8, 1760], [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1248], [4, 17, 17, 128], [4, 17, 17, 1248], [4, 17, 17, 224], [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1216], [4, 17, 17, 1216], [4, 17, 17, 224], [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1152], [4, 17, 17, 1152], [4, 17, 17, 192], [4, 17, 17, 160], [4, 17, 17, 1152], [4, 17, 17, 1024], [4, 17, 17, 128], [4, 17, 17, 1024], [4, 17, 17, 128], [4, 17, 17, 1024], [4, 17, 17, 128], [4, 17, 17, 768], [4, 17, 17, 128], [4, 17, 17, 128], [4, 17, 17, 768], [4, 17, 17, 768], [4, 35, 35, 96], [4, 35, 35, 288], [4, 35, 35, 64], [4, 35, 35, 288], [4, 35, 35, 256], [4, 35, 35, 48], [4, 35, 35, 256], [4, 35, 35, 96], [4, 35, 35, 192], [4, 35, 35, 192], [4, 35, 35, 192], [4, 73, 73, 64], [4, 73, 73, 64], [4, 147, 147, 24]]
    filter_sizes = [[1, 1, 1248, 128], [1, 3, 384, 384], [3, 1, 384, 384], [1, 1, 2048, 192], [3, 3, 448, 384], [1, 1, 2048, 320], [1, 1, 2048, 448], [1, 1, 2048, 384], [1, 1, 1760, 384], [1, 1, 1760, 192], [1, 1, 1760, 448], [1, 1, 1760, 320], [3, 3, 192, 192], [3, 3, 192, 192], [1, 1, 1248, 192], [3, 3, 128, 320], [1, 1, 1248, 128], [1, 3, 224, 224], [3, 1, 192, 256], [1, 3, 192, 256], [1, 1, 1216, 192], [1, 1, 1216, 96], [3, 1, 224, 224], [3, 3, 192, 224], [1, 3, 192, 192], [1, 1, 1152, 192], [1, 1, 1152, 128], [3, 1, 192, 192], [3, 3, 160, 192], [1, 1, 1152, 160], [1, 1, 1024, 128], [1, 3, 128, 192], [1, 1, 1024, 160], [3, 1, 128, 192], [1, 1, 1024, 256], [3, 1, 128, 128], [1, 1, 768, 192], [1, 3, 128, 128], [3, 3, 128, 128], [1, 1, 768, 128], [1, 1, 768, 320], [3, 3, 96, 96], [3, 3, 288, 384], [3, 3, 64, 96], [1, 1, 288, 64], [1, 1, 256, 64], [5, 5, 48, 64], [1, 1, 256, 48], [3, 3, 96, 96], [1, 1, 192, 32], [1, 1, 192, 64], [1, 1, 192, 48], [3, 3, 64, 192], [1, 1, 64, 64], [1, 1, 24, 64]]
    out_sizes = [[4, 5, 5, 128], [4, 8, 8, 384], [4, 8, 8, 384], [4, 8, 8, 192], [4, 8, 8, 384], [4, 8, 8, 320], [4, 8, 8, 448], [4, 8, 8, 384], [4, 8, 8, 384], [4, 8, 8, 192], [4, 8, 8, 448], [4, 8, 8, 320], [4, 8, 8, 192], [4, 17, 17, 192], [4, 17, 17, 192], [4, 8, 8, 320], [4, 17, 17, 128], [4, 17, 17, 224], [4, 17, 17, 256], [4, 17, 17, 256], [4, 17, 17, 192], [4, 17, 17, 96], [4, 17, 17, 224], [4, 17, 17, 224], [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 160], [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 160], [4, 17, 17, 192], [4, 17, 17, 256], [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 128], [4, 17, 17, 128], [4, 17, 17, 128], [4, 17, 17, 320], [4, 17, 17, 96], [4, 17, 17, 384], [4, 35, 35, 96], [4, 35, 35, 64], [4, 35, 35, 64], [4, 35, 35, 64], [4, 35, 35, 48], [4, 35, 35, 96], [4, 35, 35, 32], [4, 35, 35, 64], [4, 35, 35, 48], [4, 71, 71, 192], [4, 73, 73, 64], [4, 147, 147, 64]]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in input_sizes:
        i[3] //= shrink
    for f in filter_sizes:
        f[2] //= shrink
        f[3] //= shrink
    for o in out_sizes:
        o[3] //= shrink
    VALID = 'VALID'
    SAME = 'SAME'
    paddings = [SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, VALID, SAME, SAME, VALID, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, VALID, VALID, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, VALID, VALID, VALID]
    for (i, f, o, s, p) in zip(input_sizes, filter_sizes, out_sizes, strides, paddings):
        yield (i, f, o, s, p)

def GetTestConfigs():
    if False:
        return 10
    'Get all the valid tests configs to run.\n\n  Returns:\n    all the valid test configs as tuples of data_format and use_gpu.\n  '
    test_configs = [('NHWC', False), ('NHWC', True)]
    if test.is_gpu_available(cuda_only=True):
        test_configs += [('NCHW', True)]
    return test_configs
TEST_PARAMS = [('Conv2D_NHWC_float_cpu', 'NHWC', dtypes.float32, False, 'Conv2D'), ('Conv2D_NHWC_half_cpu', 'NHWC', dtypes.float16, False, 'Conv2D'), ('Conv2D_NHWC_double_cpu', 'NHWC', dtypes.float64, False, 'Conv2D'), ('Conv2D_NHWC_bfloat16_cpu', 'NHWC', dtypes.bfloat16, False, 'Conv2D'), ('Conv2D_NHWC_int32_cpu', 'NHWC', dtypes.int32, False, 'Conv2D'), ('Conv2D_NHWC_float_gpu', 'NHWC', dtypes.float32, True, 'Conv2D'), ('Conv2D_NHWC_half_gpu', 'NHWC', dtypes.float16, True, 'Conv2D'), ('Conv2D_NHWC_double_gpu', 'NHWC', dtypes.float64, True, 'Conv2D'), ('Conv2D_NHWC_bfloat16_gpu', 'NHWC', dtypes.bfloat16, True, 'Conv2D'), ('Conv2D_NCHW_float_gpu', 'NCHW', dtypes.float32, True, 'Conv2D'), ('Conv2D_NCHW_half_gpu', 'NCHW', dtypes.float16, True, 'Conv2D'), ('Conv2D_NCHW_double_gpu', 'NCHW', dtypes.float64, True, 'Conv2D'), ('Conv2D_NCHW_bfloat16_gpu', 'NCHW', dtypes.bfloat16, True, 'Conv2D'), ('Conv_NHWC_float_cpu', 'NHWC', dtypes.float32, False, 'Conv'), ('Conv_NHWC_half_cpu', 'NHWC', dtypes.float16, False, 'Conv'), ('Conv_NHWC_double_cpu', 'NHWC', dtypes.float64, False, 'Conv'), ('Conv_NHWC_bfloat16_cpu', 'NHWC', dtypes.bfloat16, False, 'Conv'), ('Conv_NHWC_int32_cpu', 'NHWC', dtypes.int32, False, 'Conv'), ('Conv_NHWC_float_gpu', 'NHWC', dtypes.float32, True, 'Conv'), ('Conv_NHWC_half_gpu', 'NHWC', dtypes.float16, True, 'Conv'), ('Conv_NHWC_double_gpu', 'NHWC', dtypes.float64, True, 'Conv'), ('Conv_NHWC_bfloat16_gpu', 'NHWC', dtypes.bfloat16, True, 'Conv'), ('Conv_NCHW_float_gpu', 'NCHW', dtypes.float32, True, 'Conv'), ('Conv_NCHW_half_gpu', 'NCHW', dtypes.float16, True, 'Conv'), ('Conv_NCHW_double_gpu', 'NCHW', dtypes.float64, True, 'Conv'), ('Conv_NCHW_bfloat16_gpu', 'NCHW', dtypes.bfloat16, True, 'Conv')]
DILATED_PARAMS = [('Conv2D_NHWC_cpu', 'NHWC', False, 'Conv2D'), ('Conv2D_NHWC_gpu', 'NHWC', True, 'Conv2D'), ('Conv2D_NCHW_gpu', 'NCHW', True, 'Conv2D'), ('Conv_NHWC_cpu', 'NHWC', False, 'Conv'), ('Conv_NHWC_gpu', 'NHWC', True, 'Conv'), ('Conv_NCHW_gpu', 'NCHW', True, 'Conv')]

@test_util.run_all_without_tensor_float_32('Avoid TF32 conv on GPU')
class Conv2DTest(parameterized.TestCase, test.TestCase):

    def _DtypesToTest(self, use_gpu):
        if False:
            for i in range(10):
                print('nop')
        if test_util.IsMklEnabled():
            return [dtypes.float32]
        if use_gpu:
            out = [dtypes.float32]
            if test_util.GpuSupportsHalfMatMulAndConv():
                out.append(dtypes.float16)
            if not test.is_built_with_rocm():
                out.extend([dtypes.float64, dtypes.bfloat16])
            return out
        return [dtypes.float32, dtypes.float64, dtypes.float16, dtypes.bfloat16]

    def _CreateNumpyTensor(self, shape):
        if False:
            i = 10
            return i + 15
        total_size = 1
        for s in shape:
            total_size *= s
        return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

    def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, dilations, strides, padding, data_format, dtype, use_gpu, op_name):
        if False:
            i = 10
            return i + 15
        'Verifies the output values of the convolution function.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,\n        input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,\n        input_depth, output_depth].\n      dilations: Dilated rate: [col_dilation, row_dilation]\n      strides: Stride: [col_stride, row_stride]\n      padding: Padding type.\n      data_format: Format of the data tensors.\n      dtype: Data type for inputs and outputs.\n      use_gpu: True if the operations should be run on GPU\n      op_name: Name of the op to be tested\n\n    Returns:\n      Symbolic tensor value that can be used to execute the computation\n    '
        x1 = self._CreateNumpyTensor(tensor_in_sizes)
        x2 = self._CreateNumpyTensor(filter_in_sizes)
        with test_util.device(use_gpu):
            t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
            t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
            strides = [1] + strides + [1]
            dilations = [1] + dilations + [1]
            if isinstance(padding, (list, tuple)):
                padding = [(0, 0)] + padding + [(0, 0)]
            if data_format == 'NCHW':
                t1 = test_util.NHWCToNCHW(t1)
                strides = test_util.NHWCToNCHW(strides)
                dilations = test_util.NHWCToNCHW(dilations)
                if isinstance(padding, (list, tuple)):
                    padding = test_util.NHWCToNCHW(padding)
            if op_name == 'Conv2D':
                conv = nn_ops.conv2d(t1, t2, dilations=dilations, strides=strides, padding=padding, data_format=data_format)
            elif op_name == 'Conv':
                conv_format = 'CHANNELS_LAST' if data_format == 'NHWC' else 'CHANNELS_FIRST'
                (conv_padding, explicit_paddings) = nn_ops.convert_padding(padding)
                conv = gen_nn_ops.conv(t1, t2, strides=strides, padding=conv_padding, explicit_paddings=explicit_paddings, data_format=conv_format, dilations=dilations)
            else:
                raise ValueError('Invalid op name: %s' % op_name)
            self.assertEqual(conv.dtype, dtype)
            if data_format == 'NCHW':
                conv = test_util.NCHWToNHWC(conv)
            return conv

    def _CompareFwdValues(self, tensor_in_sizes, filter_in_sizes, conv_strides, padding):
        if False:
            print('Hello World!')
        'Verifies that CPU and GPU produce the same values.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in\n        [batch, input_rows, input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in\n        [kernel_rows, kernel_cols, input_depth, output_depth].\n      conv_strides: [row_stride, col_stride] for the convolution;\n      padding: Padding type.\n    '
        x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
        x2 = np.random.rand(*filter_in_sizes).astype(np.float32)

        def _SetupVal(data_format, use_gpu):
            if False:
                print('Hello World!')
            with test_util.device(use_gpu):
                t1 = constant_op.constant(x1, shape=tensor_in_sizes)
                t2 = constant_op.constant(x2, shape=filter_in_sizes)
                strides = [1] + conv_strides + [1]
                if data_format == 'NCHW':
                    t1 = test_util.NHWCToNCHW(t1)
                    strides = test_util.NHWCToNCHW(strides)
                conv = nn_ops.conv2d(t1, t2, strides=strides, padding=padding, data_format=data_format)
                if data_format == 'NCHW':
                    conv = test_util.NCHWToNHWC(conv)
                return conv
        tensors = []
        for (data_format, use_gpu) in GetTestConfigs():
            tensors.append(_SetupVal(data_format, use_gpu))
        values = self.evaluate(tensors)
        for i in range(1, len(values)):
            self.assertAllClose(values[0], values[i], rtol=0.001, atol=0.001)

    def _ComputeReferenceDilatedConv(self, tensor_in_sizes, filter_in_sizes, stride, dilation, padding, data_format, use_gpu):
        if False:
            print('Hello World!')
        x1 = self._CreateNumpyTensor(tensor_in_sizes)
        x2 = self._CreateNumpyTensor(filter_in_sizes)
        with test_util.device(use_gpu):
            t1 = constant_op.constant(x1, shape=tensor_in_sizes)
            t2 = constant_op.constant(x2, shape=filter_in_sizes)
            if isinstance(stride, collections_abc.Iterable):
                strides = list(stride)
            else:
                strides = [stride, stride]
            if data_format == 'NCHW':
                t1 = test_util.NHWCToNCHW(t1)
                full_strides = [1, 1] + strides
                full_dilation = [1, 1] + dilation
            else:
                full_strides = [1] + strides + [1]
                full_dilation = [1] + dilation + [1]
            expected = nn_ops.convolution(t1, t2, padding=padding, strides=strides, dilation_rate=dilation, data_format=data_format)
            computed = nn_ops.conv2d(t1, t2, strides=full_strides, dilations=full_dilation, padding=padding, data_format=data_format)
            if data_format == 'NCHW':
                expected = test_util.NCHWToNHWC(expected)
                computed = test_util.NCHWToNHWC(computed)
        return (expected, computed)

    def _ComputeReferenceDilatedConvParameters(self, tensor_in_sizes, filter_in_sizes, stride, dilation, padding, data_format, use_gpu, op_name):
        if False:
            for i in range(10):
                print('nop')
        x1 = self._CreateNumpyTensor(tensor_in_sizes)
        x2 = self._CreateNumpyTensor(filter_in_sizes)
        with test_util.device(use_gpu):
            t1 = constant_op.constant(x1, shape=tensor_in_sizes)
            t2 = constant_op.constant(x2, shape=filter_in_sizes)
            if isinstance(stride, collections_abc.Iterable):
                strides = list(stride)
            else:
                strides = [stride, stride]
            if data_format == 'NCHW':
                t1 = test_util.NHWCToNCHW(t1)
                full_strides = [1, 1] + strides
                full_dilation = [1, 1] + dilation
            else:
                full_strides = [1] + strides + [1]
                full_dilation = [1] + dilation + [1]
            expected = nn_ops.convolution(t1, t2, padding=padding, strides=strides, dilation_rate=dilation, data_format=data_format)
            if op_name == 'Conv2D':
                computed = nn_ops.conv2d(t1, t2, strides=full_strides, dilations=full_dilation, padding=padding, data_format=data_format)
            elif op_name == 'Conv':
                conv_format = 'CHANNELS_LAST' if data_format == 'NHWC' else 'CHANNELS_FIRST'
                (conv_padding, explicit_paddings) = nn_ops.convert_padding(padding)
                computed = gen_nn_ops.conv(t1, t2, strides=full_strides, dilations=full_dilation, padding=conv_padding, explicit_paddings=explicit_paddings, data_format=conv_format)
            else:
                raise ValueError('Invalid op name: %s' % op_name)
            if data_format == 'NCHW':
                expected = test_util.NCHWToNHWC(expected)
                computed = test_util.NCHWToNHWC(computed)
        return (expected, computed)

    def _VerifyDilatedConvValuesParameters(self, tensor_in_sizes, filter_in_sizes, strides, padding, dilations, data_format, use_gpu, op_name, rtol=0.0001):
        if False:
            while True:
                i = 10
        if use_gpu and (not test.is_gpu_available(cuda_only=True)):
            self.skipTest('GPU not available')
        expected_results = []
        computed_results = []
        (expected, computed) = self._ComputeReferenceDilatedConvParameters(tensor_in_sizes, filter_in_sizes, strides, dilations, padding, data_format, use_gpu, op_name)
        expected_results.append(expected)
        computed_results.append(computed)
        expected_values = self.evaluate(expected_results)
        computed_values = self.evaluate(computed_results)
        for (e_value, c_value) in zip(expected_values, computed_values):
            tf_logging.debug('expected = %s', e_value)
            tf_logging.debug('actual = %s', c_value)
            self.assertAllCloseAccordingToType(e_value.flatten(), c_value.flatten(), atol=1e-05, rtol=rtol)

    def _VerifyDilatedConvValues(self, tensor_in_sizes, filter_in_sizes, strides, padding, dilations, rtol=0.0001):
        if False:
            while True:
                i = 10
        expected_results = []
        computed_results = []
        for (data_format, use_gpu) in GetTestConfigs():
            (expected, computed) = self._ComputeReferenceDilatedConv(tensor_in_sizes, filter_in_sizes, strides, dilations, padding, data_format, use_gpu)
            expected_results.append(expected)
            computed_results.append(computed)
        tolerance = 0.01 if use_gpu else 1e-05
        expected_values = self.evaluate(expected_results)
        computed_values = self.evaluate(computed_results)
        for (e_value, c_value) in zip(expected_values, computed_values):
            tf_logging.debug('expected = %s', e_value)
            tf_logging.debug('actual = %s', c_value)
            self.assertAllClose(e_value.flatten(), c_value.flatten(), atol=tolerance, rtol=rtol)

    def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, strides, padding, expected, dilations=(1, 1), gpu_only=False, test_grappler_layout_optimizer=False, tol=1e-05):
        if False:
            for i in range(10):
                print('nop')
        if gpu_only and (not test.is_gpu_available(cuda_only=True)):
            return
        tensors = []
        dilations = list(dilations)
        for (data_format, use_gpu, op_name) in GetTestConfigs():
            if gpu_only and (not use_gpu):
                continue
            dtypes_to_test = self._DtypesToTest(use_gpu)
            if not test_grappler_layout_optimizer and data_format == 'NHWC':
                dtypes_to_test.append(dtypes.int32)
            for dtype in dtypes_to_test:
                result = self._SetupValuesForDevice(tensor_in_sizes, filter_in_sizes, dilations, strides, padding, data_format, dtype, use_gpu=use_gpu, op_name=op_name)
                if test_grappler_layout_optimizer and data_format == 'NHWC' and use_gpu:
                    result = array_ops.identity(result)
                tensors.append(result)
            values = self.evaluate(tensors)
            for i in range(len(tensors)):
                conv = tensors[i]
                value = values[i]
                tf_logging.debug('expected = %s', expected)
                tf_logging.debug('actual = %s', value)
                if np.issubdtype(value.dtype, np.integer):
                    self.assertAllEqual(np.rint(expected), np.ravel(value))
                else:
                    self.assertAllCloseAccordingToType(expected, np.ravel(value), atol=tol, rtol=tol)
                self.assertShapeEqual(value, conv)
                self.assertEqual(value.dtype, conv.dtype.as_numpy_dtype)

    def _VerifyValuesParameters(self, tensor_in_sizes, filter_in_sizes, strides, padding, expected, data_format, dtype, use_gpu, op_name, dilations=(1, 1), gpu_only=False, test_grappler_layout_optimizer=False, tol=1e-05):
        if False:
            while True:
                i = 10
        if gpu_only and (not use_gpu) or not test.is_gpu_available(cuda_only=True):
            self.skipTest('GPU not available')
        if (test_grappler_layout_optimizer or data_format != 'NHWC') and dtype == dtypes.int32:
            self.skipTest('int32 not supported')
        tensors = []
        dilations = list(dilations)
        result = self._SetupValuesForDevice(tensor_in_sizes, filter_in_sizes, dilations, strides, padding, data_format, dtype, use_gpu=use_gpu, op_name=op_name)
        if test_grappler_layout_optimizer and data_format == 'NHWC' and use_gpu:
            result = array_ops.identity(result)
        tensors.append(result)
        values = self.evaluate(tensors)
        for i in range(len(tensors)):
            conv = tensors[i]
            value = values[i]
            tf_logging.debug('expected = %s', expected)
            tf_logging.debug('actual = %s', value)
            if np.issubdtype(value.dtype, np.integer):
                self.assertAllEqual(np.rint(expected), np.ravel(value))
            else:
                self.assertAllCloseAccordingToType(expected, np.ravel(value), atol=tol, rtol=tol)
            self.assertShapeEqual(value, conv)
            self.assertEqual(value.dtype, conv.dtype.as_numpy_dtype)

    def _VerifyExplicitPaddings(self, tensor_in_sizes, filter_in_sizes, strides, padding, data_format, dtype, use_gpu, op_name, dilations=(1, 1), test_grappler_layout_optimizer=False, tol=1e-05):
        if False:
            for i in range(10):
                print('nop')
        'Verifies Conv2D with explicit padding generates correct values.\n\n    It does this by comparing with Conv2D without explicit padding. This\n    function assumes Conv2D without explicit padding works correctly.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,\n        input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,\n        input_depth, output_depth].\n      strides: [row_stride, col_stride] for the convolution;\n      padding: Explicit padding amounts.\n      data_format: "NCHW" or "NHWC"\n      dtype: data type to perform test\n      use_gpu: True if testing on the GPU\n      op_name: "Conv" or "Conv2D"\n      dilations: Dilation values\n      test_grappler_layout_optimizer: If True, allow the Grappler layout\n        optimizer to run, which turns NHWC Conv2Ds on the GPU to NCHW Conv2Ds.\n      tol: The absolute and relative tolerance.\n    '
        input_tensor = self._CreateNumpyTensor(tensor_in_sizes)
        filter_tensor = self._CreateNumpyTensor(filter_in_sizes)
        input_tensor = array_ops.pad(input_tensor, [(0, 0)] + padding + [(0, 0)])
        dilations = list(dilations)
        conv2d_result = nn_ops.conv2d(input_tensor, filter_tensor, [1] + list(strides) + [1], 'VALID', dilations=[1] + dilations + [1])
        expected = list(self.evaluate(array_ops.reshape(conv2d_result, [-1])))
        self._VerifyValuesParameters(tensor_in_sizes, filter_in_sizes, strides, padding, expected, data_format, dtype, use_gpu, op_name, dilations, test_grappler_layout_optimizer=test_grappler_layout_optimizer, tol=tol)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2D1x1Filter(self, data_format, dtype, use_gpu, op_name):
        if False:
            for i in range(10):
                print('nop')
        expected_output = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0]
        self._VerifyValuesParameters(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[1, 1, 3, 3], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @test_util.run_in_graph_and_eager_modes
    def testConv2DExpandedBatch(self):
        if False:
            for i in range(10):
                print('nop')
        tensor_in_sizes_batch = [10, 2, 3, 3]
        tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 3]
        filter_in_sizes = [1, 1, 3, 3]
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
        x2 = x1.reshape(tensor_in_sizes_expanded_batch)
        conv1 = nn_ops.conv2d(x1, filter_in, strides=[1, 1], padding='VALID')
        conv2 = nn_ops.conv2d(x2, filter_in, strides=[1, 1], padding='VALID')
        self.assertEqual(conv1.shape, tensor_in_sizes_batch)
        self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
        self.assertAllEqual(conv1, self.evaluate(conv2).reshape(conv1.shape))

    @test_util.run_in_graph_and_eager_modes
    def testConvExpandedBatch(self):
        if False:
            print('Hello World!')
        tensor_in_sizes_batch = [10, 2, 3, 3]
        tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 3]
        batch_dims = 2
        filter_in_sizes = [1, 1, 3, 3]
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
        x2 = x1.reshape(tensor_in_sizes_expanded_batch)
        conv1 = gen_nn_ops.conv(x1, filter_in, strides=[1, 1, 1, 1], padding='VALID')
        conv2 = gen_nn_ops.conv(x2, filter_in, strides=[1, 1, 1, 1], padding='VALID', batch_dims=batch_dims)
        self.assertEqual(conv1.shape, tensor_in_sizes_batch)
        self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
        self.assertAllEqual(conv1, self.evaluate(conv2).reshape(conv1.shape))

    @test_util.run_in_graph_and_eager_modes
    def testConvolutionClass2DExpandedBatch(self):
        if False:
            for i in range(10):
                print('nop')
        tensor_in_sizes_batch = [10, 2, 3, 3]
        tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 3]
        filter_in_sizes = [1, 1, 3, 3]
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
        x2 = x1.reshape(tensor_in_sizes_expanded_batch)
        convolver1 = nn_ops.Convolution(input_shape=x1.shape, filter_shape=filter_in.shape, strides=[1, 1], padding='VALID')
        self.assertEqual(convolver1.num_batch_dims, 1)
        convolver2 = nn_ops.Convolution(input_shape=x2.shape, filter_shape=filter_in.shape, strides=[1, 1], padding='VALID')
        self.assertEqual(convolver2.num_batch_dims, 2)
        conv1 = convolver1(x1, filter_in)
        conv2 = convolver2(x2, filter_in)
        self.assertEqual(conv1.shape, tensor_in_sizes_batch)
        self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
        self.assertAllEqual(conv1, self.evaluate(conv2).reshape(conv1.shape))

    @test_util.run_in_graph_and_eager_modes
    def testConvolutionWith2SpatialDimensionsAndExpandedBatch(self):
        if False:
            while True:
                i = 10
        tensor_in_sizes_batch = [10, 2, 3, 3]
        tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 3]
        filter_in_sizes = [1, 1, 3, 3]
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
        x2 = x1.reshape(tensor_in_sizes_expanded_batch)
        conv1 = nn_ops.convolution(x1, filter_in, strides=[1, 1], padding='VALID')
        conv2 = nn_ops.convolution(x2, filter_in, strides=[1, 1], padding='VALID')
        self.assertEqual(conv1.shape, tensor_in_sizes_batch)
        self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
        self.assertAllEqual(conv1, self.evaluate(conv2).reshape(conv1.shape))

    @parameterized.named_parameters(*DILATED_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2Filter2x1Dilation(self, data_format, use_gpu, op_name):
        if False:
            i = 10
            return i + 15
        self._VerifyDilatedConvValuesParameters(tensor_in_sizes=[1, 4, 4, 1], filter_in_sizes=[2, 2, 1, 1], strides=[1, 1], dilations=[2, 1], padding='VALID', data_format=data_format, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2DEmpty(self, data_format, dtype, use_gpu, op_name):
        if False:
            i = 10
            return i + 15
        expected_output = []
        self._VerifyValuesParameters(tensor_in_sizes=[0, 2, 3, 3], filter_in_sizes=[1, 1, 3, 3], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*DILATED_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2DEmptyDilation(self, data_format, use_gpu, op_name):
        if False:
            for i in range(10):
                print('nop')
        self._VerifyDilatedConvValuesParameters(tensor_in_sizes=[0, 2, 3, 3], filter_in_sizes=[1, 1, 3, 3], strides=[1, 1], dilations=[2, 1], padding='VALID', data_format=data_format, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2Filter(self, data_format, dtype, use_gpu, op_name):
        if False:
            i = 10
            return i + 15
        expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
        self._VerifyValuesParameters(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 3], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*DILATED_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2FilterDilation(self, data_format, use_gpu, op_name):
        if False:
            while True:
                i = 10
        self._VerifyDilatedConvValuesParameters(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 3], strides=[1, 1], dilations=[1, 2], padding='VALID', data_format=data_format, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2D1x2Filter(self, data_format, dtype, use_gpu, op_name):
        if False:
            while True:
                i = 10
        expected_output = [231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0, 936.0, 1029.0]
        self._VerifyValuesParameters(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[1, 2, 3, 3], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*DILATED_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2D1x2FilterDilation(self, data_format, use_gpu, op_name):
        if False:
            return 10
        self._VerifyDilatedConvValuesParameters(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[1, 2, 3, 3], strides=[1, 1], dilations=[2, 1], padding='VALID', data_format=data_format, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2FilterStride2(self, data_format, dtype, use_gpu, op_name):
        if False:
            return 10
        expected_output = [2271.0, 2367.0, 2463.0]
        self._VerifyValuesParameters(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 3], strides=[2, 2], padding='VALID', expected=expected_output, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2FilterStride2Same(self, data_format, dtype, use_gpu, op_name):
        if False:
            i = 10
            return i + 15
        expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
        self._VerifyValuesParameters(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 3], strides=[2, 2], padding='SAME', expected=expected_output, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2FilterStride1x2(self, data_format, dtype, use_gpu, op_name):
        if False:
            while True:
                i = 10
        expected_output = [58.0, 78.0, 98.0, 118.0, 138.0, 158.0]
        self._VerifyValuesParameters(tensor_in_sizes=[1, 3, 6, 1], filter_in_sizes=[2, 2, 1, 1], strides=[1, 2], padding='VALID', expected=expected_output, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2DKernelSmallerThanStrideValid(self, data_format, dtype, use_gpu, op_name):
        if False:
            for i in range(10):
                print('nop')
        expected_output = [65, 95, 275, 305]
        self._VerifyValuesParameters(tensor_in_sizes=[1, 7, 7, 1], filter_in_sizes=[2, 2, 1, 1], strides=[3, 3], padding='VALID', expected=expected_output, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2DKernelSmallerThanStrideSame(self, data_format, dtype, use_gpu, op_name):
        if False:
            while True:
                i = 10
        self._VerifyValuesParameters(tensor_in_sizes=[1, 3, 3, 1], filter_in_sizes=[1, 1, 1, 1], strides=[2, 2], padding='SAME', expected=[1, 3, 7, 9], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyValuesParameters(tensor_in_sizes=[1, 4, 4, 1], filter_in_sizes=[1, 1, 1, 1], strides=[2, 2], padding='SAME', expected=[1, 3, 9, 11], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyValuesParameters(tensor_in_sizes=[1, 4, 4, 1], filter_in_sizes=[2, 2, 1, 1], strides=[3, 3], padding='SAME', expected=[44, 28, 41, 16], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2DKernelSizeMatchesInputSize(self, data_format, dtype, use_gpu, op_name):
        if False:
            i = 10
            return i + 15
        self._VerifyValuesParameters(tensor_in_sizes=[1, 2, 2, 1], filter_in_sizes=[2, 2, 1, 2], strides=[1, 1], padding='VALID', expected=[50, 60], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*DILATED_PARAMS)
    @test_util.run_in_graph_and_eager_modes
    def testConv2DKernelSizeMatchesInputSizeDilation(self, data_format, use_gpu, op_name):
        if False:
            return 10
        self._VerifyDilatedConvValuesParameters(tensor_in_sizes=[1, 3, 3, 1], filter_in_sizes=[2, 2, 1, 2], strides=[1, 1], dilations=[2, 2], padding='VALID', data_format=data_format, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes()
    def testConv2D0x0Padding(self, data_format, dtype, use_gpu, op_name):
        if False:
            print('Hello World!')
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 3], strides=[1, 1], padding=[[0, 0], [0, 0]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyExplicitPaddings(tensor_in_sizes=[3, 4, 3, 2], filter_in_sizes=[1, 1, 2, 1], strides=[2, 2], padding=[[0, 0], [0, 0]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes()
    def testConv2D1x1Padding(self, data_format, dtype, use_gpu, op_name):
        if False:
            return 10
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 3, 2], filter_in_sizes=[2, 2, 2, 2], strides=[1, 1], padding=[[1, 1], [1, 1]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 2, 1], filter_in_sizes=[1, 1, 1, 2], strides=[1, 1], padding=[[1, 1], [1, 1]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Padding(self, data_format, dtype, use_gpu, op_name):
        if False:
            print('Hello World!')
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 1, 2], filter_in_sizes=[2, 1, 2, 1], strides=[1, 1], padding=[[2, 2], [2, 2]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 1, 2], filter_in_sizes=[1, 1, 2, 1], strides=[2, 1], padding=[[2, 2], [2, 2]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes()
    def testConv2DOnlyBottomPadding(self, data_format, dtype, use_gpu, op_name):
        if False:
            for i in range(10):
                print('nop')
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 2], strides=[1, 1], padding=[[0, 3], [0, 0]], tol=2e-05, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyExplicitPaddings(tensor_in_sizes=[2, 2, 4, 3], filter_in_sizes=[1, 2, 3, 2], strides=[2, 2], padding=[[0, 3], [0, 0]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes()
    def testConv2DOnlyTopRightPadding(self, data_format, dtype, use_gpu, op_name):
        if False:
            i = 10
            return i + 15
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 3, 3], filter_in_sizes=[2, 2, 3, 2], strides=[1, 1], padding=[[1, 0], [0, 2]], tol=5e-05, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 4, 2], filter_in_sizes=[2, 2, 2, 2], strides=[1, 3], padding=[[1, 0], [0, 2]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes()
    def testConv2DLotsPadding(self, data_format, dtype, use_gpu, op_name):
        if False:
            for i in range(10):
                print('nop')
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 1, 1, 3], filter_in_sizes=[2, 2, 3, 3], strides=[1, 1], padding=[[3, 4], [4, 2]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 1, 1], filter_in_sizes=[2, 2, 1, 3], strides=[2, 1], padding=[[3, 4], [4, 2]], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes()
    def testConv2DExplicitPaddingWithDilations(self, data_format, dtype, use_gpu, op_name):
        if False:
            for i in range(10):
                print('nop')
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 3, 2, 1], filter_in_sizes=[1, 2, 1, 2], strides=[1, 1], padding=[[1, 0], [0, 1]], dilations=[2, 1], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 3, 2], filter_in_sizes=[3, 2, 2, 1], strides=[1, 1], padding=[[2, 1], [1, 2]], dilations=[2, 3], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    @test_util.run_in_graph_and_eager_modes()
    def testConv2dOnlyPaddingReturnsZeros(self, data_format, dtype, use_gpu, op_name):
        if False:
            while True:
                i = 10
        self._VerifyValuesParameters(tensor_in_sizes=[1, 0, 2, 1], filter_in_sizes=[1, 1, 1, 1], strides=[1, 1], padding=[[1, 1], [1, 1]], expected=[0, 0, 0, 0, 0, 0, 0, 0], data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    @parameterized.named_parameters(*TEST_PARAMS)
    def testConv2DExplicitPaddingWithLayoutOptimizer(self, data_format, dtype, use_gpu, op_name):
        if False:
            for i in range(10):
                print('nop')
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 3, 2, 1], filter_in_sizes=[1, 2, 1, 2], strides=[1, 1], padding=[[1, 0], [0, 1]], dilations=[2, 1], test_grappler_layout_optimizer=True, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)
        self._VerifyExplicitPaddings(tensor_in_sizes=[1, 2, 3, 2], filter_in_sizes=[3, 2, 2, 1], strides=[1, 1], padding=[[2, 1], [1, 2]], dilations=[2, 3], test_grappler_layout_optimizer=True, data_format=data_format, dtype=dtype, use_gpu=use_gpu, op_name=op_name)

    def _VerifyGroupConvFwd(self, tensor_in_sizes, filter_in_sizes, dilations, strides, padding, data_format, dtype):
        if False:
            for i in range(10):
                print('nop')
        'Verify the output of group convolution is equal to a for-loop implementation.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,\n        input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,\n        input_depth, output_depth].\n      dilations: Dilated rate: [col_dilation, row_dilation]\n      strides: Stride: [col_stride, row_stride]\n      padding: Padding type.\n      data_format: Format of the data tensors.\n      dtype: Data type for inputs and outputs.\n    '
        tensor_in = self._CreateNumpyTensor(tensor_in_sizes)
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        num_groups = tensor_in_sizes[3] // filter_in_sizes[2]
        assert num_groups > 1 and filter_in_sizes[2] * num_groups == tensor_in_sizes[3]
        with test_util.device(True):
            t1 = constant_op.constant(tensor_in, dtype=dtype)
            t2 = constant_op.constant(filter_in, dtype=dtype)
            strides = [1] + strides + [1]
            dilations = [1] + dilations + [1]
            if data_format == 'NCHW':
                t1 = test_util.NHWCToNCHW(t1)
                strides = test_util.NHWCToNCHW(strides)
                dilations = test_util.NHWCToNCHW(dilations)
                t1_splits = array_ops.split(t1, num_groups, axis=1)
            else:
                t1_splits = array_ops.split(t1, num_groups, axis=3)
            t2_splits = array_ops.split(t2, num_groups, axis=3)

            def MakeConv2d(inputs, filters):
                if False:
                    while True:
                        i = 10
                return nn_ops.conv2d(inputs, filters, strides, padding, dilations=dilations, data_format=data_format)
            group_conv = MakeConv2d(t1, t2)
            group_conv_loop = array_ops.concat([MakeConv2d(t1s, t2s) for (t1s, t2s) in zip(t1_splits, t2_splits)], axis=1 if data_format == 'NCHW' else 3)
            results = self.evaluate([group_conv, group_conv_loop])
            tol_to_use = 1e-05
            self.assertAllClose(results[0], results[1], atol=tol_to_use, rtol=tol_to_use)

    @test_util.run_in_graph_and_eager_modes
    def testConv2DGroupConvFwd(self):
        if False:
            for i in range(10):
                print('nop')
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            data_formats = ['NHWC', 'NCHW']
        else:
            data_formats = ['NHWC']
        for data_format in data_formats:
            for dilation in [1, 2]:
                for stride in [1, 2]:
                    for filter_dims in [[3, 3, 4, 8], [1, 1, 2, 16]]:
                        self._VerifyGroupConvFwd([10, 32, 32, 16], filter_dims, dilations=[dilation, dilation], strides=[stride, stride], padding='SAME', data_format=data_format, dtype=dtypes.float32)

    @test_util.deprecated_graph_mode_only
    @test_util.run_cuda_only
    def testInputGradientGroupConv(self):
        if False:
            i = 10
            return i + 15
        for data_format in ['NCHW', 'NHWC']:
            for test_input in [True, False]:
                self.ConstructAndTestGradient(batch=2, input_rows=5, input_cols=4, filter_rows=3, filter_cols=3, num_groups=2, padding='VALID', in_depth=4, out_depth=6, stride_rows=1, stride_cols=1, test_input=test_input, data_format=data_format, use_gpu=True, max_err=0.005)

    @test_util.deprecated_graph_mode_only
    @test_util.run_cuda_only
    def testFilterGradientGroupConv(self):
        if False:
            return 10
        for data_format in ['NCHW', 'NHWC']:
            for test_input in [True, False]:
                self.ConstructAndTestGradient(batch=2, input_rows=5, input_cols=4, filter_rows=3, filter_cols=3, num_groups=2, padding='VALID', in_depth=4, out_depth=6, stride_rows=1, stride_cols=1, test_input=test_input, data_format=data_format, use_gpu=True, max_err=0.005)

    def _RunAndVerifyBackpropInput(self, input_sizes, filter_sizes, output_sizes, strides, padding, expected, data_format, use_gpu, err, dilations=(1, 1)):
        if False:
            for i in range(10):
                print('nop')
        if use_gpu and (not test.is_gpu_available(cuda_only=True)):
            return
        x1 = self._CreateNumpyTensor(filter_sizes)
        x2 = self._CreateNumpyTensor(output_sizes)
        dilations = list(dilations)
        with test_util.device(use_gpu):
            if len(input_sizes) == 4:
                if data_format == 'NCHW':
                    input_sizes = test_util.NHWCToNCHW(input_sizes)
            t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
            t1 = constant_op.constant(x1, shape=filter_sizes)
            t2 = constant_op.constant(x2, shape=output_sizes)
            strides = [1] + strides + [1]
            dilations = [1] + dilations + [1]
            if isinstance(padding, (list, tuple)):
                padding = [(0, 0)] + padding + [(0, 0)]
            if data_format == 'NCHW':
                t2 = test_util.NHWCToNCHW(t2)
                strides = test_util.NHWCToNCHW(strides)
                dilations = test_util.NHWCToNCHW(dilations)
                if isinstance(padding, (list, tuple)):
                    padding = test_util.NHWCToNCHW(padding)
            conv = nn_ops.conv2d_backprop_input(t0, t1, t2, strides=strides, padding=padding, data_format=data_format, dilations=dilations)
            if data_format == 'NCHW':
                conv = test_util.NCHWToNHWC(conv)
            value = self.evaluate(conv)
            self.assertShapeEqual(value, conv)
        tf_logging.debug('expected = %s', expected)
        tf_logging.debug('actual = %s', value)
        self.assertAllCloseAccordingToType(expected, value.flatten(), atol=1e-05)

    def _CompareBackpropInput(self, input_sizes, filter_sizes, output_sizes, conv_strides, padding):
        if False:
            return 10
        x1 = np.random.rand(*filter_sizes).astype(np.float32)
        x2 = np.random.rand(*output_sizes).astype(np.float32)

        def _GetVal(data_format, use_gpu):
            if False:
                return 10
            with test_util.device(use_gpu):
                if data_format == 'NCHW':
                    new_input_sizes = test_util.NHWCToNCHW(input_sizes)
                else:
                    new_input_sizes = input_sizes
                t0 = constant_op.constant(new_input_sizes, shape=[len(new_input_sizes)])
                t1 = constant_op.constant(x1, shape=filter_sizes)
                t2 = constant_op.constant(x2, shape=output_sizes)
                strides = [1] + conv_strides + [1]
                if data_format == 'NCHW':
                    t2 = test_util.NHWCToNCHW(t2)
                    strides = test_util.NHWCToNCHW(strides)
                conv = nn_ops.conv2d_backprop_input(t0, t1, t2, strides=strides, padding=padding, data_format=data_format)
                if data_format == 'NCHW':
                    conv = test_util.NCHWToNHWC(conv)
                ret = self.evaluate(conv)
                self.assertShapeEqual(ret, conv)
                return ret
        values = []
        for (data_format, use_gpu) in GetTestConfigs():
            values.append(_GetVal(data_format, use_gpu))
        for i in range(1, len(values)):
            self.assertAllClose(values[0], values[i], rtol=0.01, atol=0.01)

    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2Depth1ValidBackpropInput(self):
        if False:
            while True:
                i = 10
        expected_output = [1.0, 4.0, 4.0, 3.0, 10.0, 8.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 1, 2, 1], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.run_in_graph_and_eager_modes
    def testConv2DEmptyBackpropInput(self):
        if False:
            print('Hello World!')
        expected_output = []
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[0, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[0, 1, 2, 1], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2Depth3ValidBackpropInput(self):
        if False:
            while True:
                i = 10
        expected_output = [14.0, 32.0, 50.0, 100.0, 163.0, 226.0, 167.0, 212.0, 257.0, 122.0, 140.0, 158.0, 478.0, 541.0, 604.0, 437.0, 482.0, 527.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 2, 3, 3], filter_sizes=[2, 2, 3, 3], output_sizes=[1, 1, 2, 3], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, use_gpu=use_gpu, err=0.0001)

    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2Depth3ValidBackpropInputStride1x2(self):
        if False:
            for i in range(10):
                print('nop')
        expected_output = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 7.0, 12.0, 11.0, 18.0, 15.0, 24.0, 12.0, 16.0, 15.0, 20.0, 18.0, 24.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 3, 6, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 2, 3, 1], strides=[1, 2], padding='VALID', expected=expected_output, data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.run_in_graph_and_eager_modes
    def testConv2DStrideTwoFilterOneSameBackpropInput(self):
        if False:
            for i in range(10):
                print('nop')
        expected_output = [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 4, 4, 1], filter_sizes=[1, 1, 1, 1], output_sizes=[1, 2, 2, 1], strides=[2, 2], padding='SAME', expected=expected_output, data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.run_in_graph_and_eager_modes
    def testConv2DKernelSizeMatchesInputSizeBackpropInput(self):
        if False:
            while True:
                i = 10
        expected_output = [5.0, 11.0, 17.0, 23.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 2, 2, 1], filter_sizes=[2, 2, 1, 2], output_sizes=[1, 1, 1, 2], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.run_in_graph_and_eager_modes
    @test_util.disable_xla('XLA requires input_sizes to be a 4D shape.')
    def testConv2DInputSizesContainsOnlySpatialDimensionsBackpropInput(self):
        if False:
            while True:
                i = 10
        expected_output = [5.0, 11.0, 17.0, 23.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[2, 2], filter_sizes=[2, 2, 1, 2], output_sizes=[1, 1, 1, 2], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.run_in_graph_and_eager_modes
    @test_util.disable_xla('b/239598470')
    def testConv2DBackpropInputDegenerateBackpropInput(self):
        if False:
            while True:
                i = 10
        input_sizes = [3, 1, 1, 2]
        expected_output = np.zeros(input_sizes).flatten()
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=input_sizes, filter_sizes=[1, 3, 2, 3], output_sizes=[3, 1, 0, 3], strides=[1, 2], padding='VALID', expected=expected_output, data_format=data_format, use_gpu=use_gpu, err=1e-05)

    def _RunAndVerifyBackpropFilter(self, input_sizes, filter_sizes, output_sizes, strides, padding, expected, data_format, use_gpu, dilations=(1, 1), err=1e-05):
        if False:
            print('Hello World!')
        x0 = self._CreateNumpyTensor(input_sizes)
        x2 = self._CreateNumpyTensor(output_sizes)
        dilations = list(dilations)
        explicit_strides = [1] + strides + [1]
        new_padding = padding
        new_dilations = [1] + dilations + [1]
        if isinstance(new_padding, (list, tuple)):
            new_padding = [(0, 0)] + new_padding + [(0, 0)]
        if data_format == 'NCHW':
            explicit_strides = test_util.NHWCToNCHW(explicit_strides)
            new_dilations = test_util.NHWCToNCHW(new_dilations)
            if isinstance(padding, (list, tuple)):
                new_padding = test_util.NHWCToNCHW(new_padding)
        for dtype in self._DtypesToTest(use_gpu=use_gpu):
            with test_util.device(use_gpu):
                t0 = constant_op.constant(x0, shape=input_sizes, dtype=dtype)
                t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
                t2 = constant_op.constant(x2, shape=output_sizes, dtype=dtype)
                if data_format == 'NCHW':
                    t0 = test_util.NHWCToNCHW(t0)
                    t2 = test_util.NHWCToNCHW(t2)
                conv = nn_ops.conv2d_backprop_filter(t0, t1, t2, strides=explicit_strides, padding=new_padding, dilations=new_dilations, data_format=data_format)
                value = self.evaluate(conv)
                self.assertShapeEqual(value, conv)
            tf_logging.debug('expected = %s', expected)
            tf_logging.debug('actual = %s', value)
            self.assertAllCloseAccordingToType(expected, value.flatten(), err)

    def _CompareBackFilter(self, input_sizes, filter_sizes, output_sizes, conv_strides, padding):
        if False:
            i = 10
            return i + 15
        x0 = np.random.rand(*input_sizes).astype(np.float32)
        x2 = np.random.rand(*output_sizes).astype(np.float32)

        def _GetVal(data_format, use_gpu):
            if False:
                i = 10
                return i + 15
            with test_util.device(use_gpu):
                t0 = constant_op.constant(x0, shape=input_sizes)
                t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
                t2 = constant_op.constant(x2, shape=output_sizes)
                strides = [1] + conv_strides + [1]
                if data_format == 'NCHW':
                    t0 = test_util.NHWCToNCHW(t0)
                    t2 = test_util.NHWCToNCHW(t2)
                    strides = test_util.NHWCToNCHW(strides)
                conv = nn_ops.conv2d_backprop_filter(t0, t1, t2, strides=strides, padding=padding, data_format=data_format)
                ret = self.evaluate(conv)
                self.assertShapeEqual(ret, conv)
                return ret
        values = []
        for (data_format, use_gpu) in GetTestConfigs():
            values.append(_GetVal(data_format, use_gpu))
        for i in range(1, len(values)):
            self.assertAllClose(values[0], values[i], rtol=0.0002, atol=0.0002)

    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2Depth1ValidBackpropFilter(self):
        if False:
            return 10
        expected = [5.0, 8.0, 14.0, 17.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 1, 2, 1], strides=[1, 1], padding='VALID', expected=expected, data_format=data_format, use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes
    def testConv2DEmptyBackpropFilter(self):
        if False:
            return 10
        expected = []
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 0], output_sizes=[1, 1, 2, 0], strides=[1, 1], padding='VALID', expected=expected, data_format=data_format, use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes
    def testConv2DBackpropFilterWithEmptyInput(self):
        if False:
            for i in range(10):
                print('nop')
        expected = [0, 0, 0, 0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[0, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[0, 1, 2, 1], strides=[1, 1], padding='VALID', expected=expected, data_format=data_format, use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2Depth3ValidBackpropFilter(self):
        if False:
            return 10
        expected = [17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0, 32.0, 43.0, 54.0, 37.0, 50.0, 63.0, 42.0, 57.0, 72.0, 62.0, 85.0, 108.0, 67.0, 92.0, 117.0, 72.0, 99.0, 126.0, 77.0, 106.0, 135.0, 82.0, 113.0, 144.0, 87.0, 120.0, 153.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 2, 3, 3], filter_sizes=[2, 2, 3, 3], output_sizes=[1, 1, 2, 3], strides=[1, 1], padding='VALID', expected=expected, data_format=data_format, use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes
    def testConv2D2x2Depth3ValidBackpropFilterStride1x2(self):
        if False:
            return 10
        expected = [161.0, 182.0, 287.0, 308.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 3, 6, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 2, 3, 1], strides=[1, 2], padding='VALID', expected=expected, data_format=data_format, use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes
    def testConv2DStrideTwoFilterOneSameBackpropFilter(self):
        if False:
            print('Hello World!')
        expected_output = [78.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 4, 4, 1], filter_sizes=[1, 1, 1, 1], output_sizes=[1, 2, 2, 1], strides=[2, 2], padding='SAME', expected=expected_output, data_format=data_format, use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes
    def testConv2DKernelSizeMatchesInputSizeBackpropFilter(self):
        if False:
            print('Hello World!')
        expected_output = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 2, 2, 1], filter_sizes=[2, 2, 1, 2], output_sizes=[1, 1, 1, 2], strides=[1, 1], padding='VALID', expected=expected_output, data_format=data_format, use_gpu=use_gpu)

    def _RunAndVerifyBackpropInputDilation(self, input_sizes, filter_sizes, output_sizes, strides, dilations, padding, data_format, use_gpu, err):
        if False:
            for i in range(10):
                print('nop')
        x1 = self._CreateNumpyTensor(input_sizes)
        x2 = self._CreateNumpyTensor(filter_sizes)
        default_dilations = dilations[0] == 1 and dilations[1] == 1
        if default_dilations or use_gpu:
            with self.cached_session(use_gpu=use_gpu):
                if data_format == 'NCHW':
                    input_sizes = test_util.NHWCToNCHW(input_sizes)
                t1 = constant_op.constant(x1, shape=input_sizes)
                t2 = constant_op.constant(x2, shape=filter_sizes)
                full_strides = [1] + strides + [1]
                full_dilations = [1] + dilations + [1]
                if data_format == 'NCHW':
                    full_strides = test_util.NHWCToNCHW(full_strides)
                    full_dilations = test_util.NHWCToNCHW(full_dilations)
                conv_forward = nn_ops.conv2d(t1, t2, strides=full_strides, dilations=full_dilations, padding=padding, data_format=data_format)
                conv_forward_2 = nn_ops.convolution(t1, t2, padding=padding, strides=strides, dilation_rate=dilations, data_format=data_format)
                if data_format == 'NCHW':
                    conv_forward = test_util.NCHWToNHWC(conv_forward)
                    conv_forward_2 = test_util.NCHWToNHWC(conv_forward_2)
                conv = gradients_impl.gradients(conv_forward, t1)[0]
                conv_2 = gradients_impl.gradients(conv_forward_2, t1)[0]
                value = self.evaluate(conv)
                value_2 = self.evaluate(conv_2)
                self.assertShapeEqual(value, conv)
                self.assertShapeEqual(value_2, conv_2)
            tf_logging.debug('expected = %s', value_2)
            tf_logging.debug('actual = %s', value)
            self.assertArrayNear(value_2.flatten(), value.flatten(), err)

    def _RunAndVerifyBackpropFilterDilation(self, input_sizes, filter_sizes, output_sizes, strides, dilations, padding, data_format, use_gpu, err):
        if False:
            print('Hello World!')
        x1 = self._CreateNumpyTensor(input_sizes)
        x2 = self._CreateNumpyTensor(filter_sizes)
        default_dilations = dilations[0] == 1 and dilations[1] == 1
        if default_dilations or use_gpu:
            with self.cached_session(use_gpu=use_gpu):
                if data_format == 'NCHW':
                    input_sizes = test_util.NHWCToNCHW(input_sizes)
                t1 = constant_op.constant(x1, shape=input_sizes)
                t2 = constant_op.constant(x2, shape=filter_sizes)
                full_strides = [1] + strides + [1]
                full_dilations = [1] + dilations + [1]
                if data_format == 'NCHW':
                    full_strides = test_util.NHWCToNCHW(full_strides)
                    full_dilations = test_util.NHWCToNCHW(full_dilations)
                conv_forward = nn_ops.conv2d(t1, t2, strides=full_strides, dilations=full_dilations, padding=padding, data_format=data_format)
                conv_forward_2 = nn_ops.convolution(t1, t2, padding=padding, strides=strides, dilation_rate=dilations, data_format=data_format)
                if data_format == 'NCHW':
                    conv_forward = test_util.NCHWToNHWC(conv_forward)
                    conv_forward_2 = test_util.NCHWToNHWC(conv_forward_2)
                conv = gradients_impl.gradients(conv_forward, t2)[0]
                conv_2 = gradients_impl.gradients(conv_forward, t2)[0]
                value = self.evaluate(conv)
                value_2 = self.evaluate(conv_2)
                self.assertShapeEqual(value, conv)
                self.assertShapeEqual(value_2, conv_2)
            tf_logging.debug('expected = %s', value_2)
            tf_logging.debug('actual = %s', value)
            self.assertArrayNear(value_2.flatten(), value.flatten(), err)

    @test_util.deprecated_graph_mode_only
    def testConv2D2x2Depth3ValidBackpropFilterStride1x1Dilation2x1(self):
        if False:
            while True:
                i = 10
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropFilterDilation(input_sizes=[1, 3, 6, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 1, 5, 1], strides=[1, 1], dilations=[2, 1], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.deprecated_graph_mode_only
    def testConv2D2x2Depth1ValidBackpropFilterDilation1x2(self):
        if False:
            print('Hello World!')
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropFilterDilation(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 1, 2, 1], strides=[1, 1], dilations=[1, 2], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.deprecated_graph_mode_only
    def testConv2DEmptyBackpropFilterDilation1x2(self):
        if False:
            for i in range(10):
                print('nop')
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropFilterDilation(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 0], output_sizes=[1, 1, 2, 0], strides=[1, 1], dilations=[1, 2], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.deprecated_graph_mode_only
    def testConv2D2x2Depth3ValidBackpropFilterDilation2x2(self):
        if False:
            print('Hello World!')
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropFilterDilation(input_sizes=[1, 3, 4, 3], filter_sizes=[2, 2, 3, 3], output_sizes=[1, 1, 2, 3], strides=[1, 1], dilations=[2, 2], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.deprecated_graph_mode_only
    def testConv2DKernelSizeMatchesInputSizeBackpropFilterDilation2x2(self):
        if False:
            print('Hello World!')
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropFilterDilation(input_sizes=[1, 3, 3, 1], filter_sizes=[2, 2, 1, 2], output_sizes=[1, 1, 1, 2], strides=[1, 1], dilations=[2, 2], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.deprecated_graph_mode_only
    def testConv2D2x2Depth3ValidBackpropInputStride1x1Dilation2x1(self):
        if False:
            while True:
                i = 10
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropInputDilation(input_sizes=[1, 3, 6, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 1, 5, 1], strides=[1, 1], dilations=[2, 1], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.deprecated_graph_mode_only
    def testConv2D2x2Depth1ValidBackpropInputDilation1x2(self):
        if False:
            for i in range(10):
                print('nop')
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropInputDilation(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 1, 2, 1], strides=[1, 1], dilations=[1, 2], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.deprecated_graph_mode_only
    def testConv2DEmptyBackpropInputDilation1x2(self):
        if False:
            print('Hello World!')
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropInputDilation(input_sizes=[0, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[0, 1, 2, 1], strides=[1, 1], dilations=[1, 2], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=1e-05)

    @test_util.deprecated_graph_mode_only
    def testConv2D2x2Depth3ValidBackpropInputDilation2x1(self):
        if False:
            return 10
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropInputDilation(input_sizes=[1, 3, 2, 3], filter_sizes=[2, 2, 3, 3], output_sizes=[1, 1, 2, 3], strides=[1, 1], dilations=[2, 1], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=0.0001)

    @test_util.deprecated_graph_mode_only
    def testConv2DKernelSizeMatchesInputSizeBackpropInputDilation2x2(self):
        if False:
            for i in range(10):
                print('nop')
        if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
            for (data_format, use_gpu) in GetTestConfigs():
                self._RunAndVerifyBackpropInputDilation(input_sizes=[1, 3, 3, 1], filter_sizes=[2, 2, 1, 2], output_sizes=[1, 1, 1, 2], strides=[1, 1], dilations=[2, 2], padding='VALID', data_format=data_format, use_gpu=use_gpu, err=1e-05)

    def _RunAndVerifyBackpropInputExplicitPadding(self, input_sizes, filter_sizes, output_sizes, strides, padding, data_format, use_gpu, dilations=(1, 1), err=2e-05):
        if False:
            return 10
        if use_gpu and (not test.is_gpu_available(cuda_only=True)):
            return
        if not use_gpu and dilations != (1, 1):
            return
        x1 = self._CreateNumpyTensor(filter_sizes)
        x2 = self._CreateNumpyTensor(output_sizes)
        dilations = list(dilations)
        padded_input_sizes = input_sizes[:]
        padded_input_sizes[1] += padding[0][0] + padding[0][1]
        padded_input_sizes[2] += padding[1][0] + padding[1][1]
        c = nn_ops.conv2d_backprop_input(padded_input_sizes, x1, x2, strides=[1] + strides + [1], padding='VALID', dilations=[1] + dilations + [1])
        c = c[:, padding[0][0]:c.shape[1] - padding[0][1], padding[1][0]:c.shape[2] - padding[1][1], :]
        expected = list(self.evaluate(array_ops.reshape(c, [-1])))
        self._RunAndVerifyBackpropInput(input_sizes, filter_sizes, output_sizes, strides, padding, expected, data_format, use_gpu=use_gpu, err=err, dilations=dilations)

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding0x0BackpropInput(self):
        if False:
            for i in range(10):
                print('nop')
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 1, 2, 1], strides=[1, 1], padding=[[0, 0], [0, 0]], data_format=data_format, use_gpu=use_gpu)
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 3, 4, 2], filter_sizes=[2, 2, 2, 3], output_sizes=[1, 1, 2, 3], strides=[2, 2], padding=[[0, 0], [0, 0]], data_format=data_format, use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding1x1BackpropInput(self):
        if False:
            while True:
                i = 10
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 2], output_sizes=[1, 3, 4, 2], strides=[1, 1], padding=[[1, 1], [1, 1]], data_format=data_format, use_gpu=use_gpu, err=0.0001)
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 2, 3, 2], filter_sizes=[1, 1, 2, 1], output_sizes=[1, 4, 3, 1], strides=[1, 2], padding=[[1, 1], [1, 1]], data_format=data_format, use_gpu=use_gpu)
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 4, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 4, 2, 1], strides=[1, 2], padding=[[1, 1], [1, 1]], data_format=data_format, dilations=[2, 2], use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding2x2BackpropInput(self):
        if False:
            while True:
                i = 10
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[2, 3, 1, 1], filter_sizes=[2, 1, 1, 1], output_sizes=[2, 2, 5, 1], strides=[3, 1], padding=[[2, 2], [2, 2]], data_format=data_format, use_gpu=use_gpu)
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 3, 6, 1], filter_sizes=[3, 2, 1, 1], output_sizes=[1, 3, 4, 1], strides=[1, 2], padding=[[2, 2], [2, 2]], data_format=data_format, dilations=[2, 3], use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding_1_8_4_1_BackpropInput(self):
        if False:
            while True:
                i = 10
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 10, 8, 1], strides=[1, 1], padding=[[1, 8], [4, 2]], data_format=data_format, use_gpu=use_gpu, err=5e-05)
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 5, 3, 1], filter_sizes=[3, 2, 1, 1], output_sizes=[1, 4, 8, 1], strides=[3, 1], padding=[[1, 8], [4, 2]], data_format=data_format, use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding_5_0_2_2_BackpropInput(self):
        if False:
            print('Hello World!')
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 3, 3, 1], filter_sizes=[2, 1, 1, 1], output_sizes=[1, 7, 7, 1], strides=[1, 1], padding=[[5, 0], [2, 2]], data_format=data_format, err=5e-05, use_gpu=use_gpu)
            self._RunAndVerifyBackpropInputExplicitPadding(input_sizes=[1, 4, 2, 1], filter_sizes=[3, 3, 1, 1], output_sizes=[1, 5, 2, 1], strides=[1, 2], padding=[[5, 0], [2, 2]], data_format=data_format, dilations=[2, 1], use_gpu=use_gpu)

    def _RunAndVerifyBackpropFilterExplicitPadding(self, input_sizes, filter_sizes, output_sizes, strides, padding, data_format, use_gpu, dilations=(1, 1), err=1e-05):
        if False:
            i = 10
            return i + 15
        if use_gpu and (not test.is_gpu_available(cuda_only=True)):
            return
        if not use_gpu and dilations != (1, 1):
            return
        x0 = self._CreateNumpyTensor(input_sizes)
        x2 = self._CreateNumpyTensor(output_sizes)
        dilations = list(dilations)
        x0 = np.pad(x0, [(0, 0)] + padding + [(0, 0)], 'constant')
        c = nn_ops.conv2d_backprop_filter(x0, filter_sizes, x2, strides=[1] + strides + [1], padding='VALID', dilations=[1] + dilations + [1])
        expected = list(self.evaluate(array_ops.reshape(c, [-1])))
        self._RunAndVerifyBackpropFilter(input_sizes, filter_sizes, output_sizes, strides, padding, expected, data_format, use_gpu=use_gpu, dilations=dilations, err=err)

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding0x0BackpropFilter(self):
        if False:
            for i in range(10):
                print('nop')
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 1, 2, 1], strides=[1, 1], padding=[[0, 0], [0, 0]], data_format=data_format, use_gpu=use_gpu)
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 3, 4, 2], filter_sizes=[2, 2, 2, 3], output_sizes=[1, 1, 2, 3], strides=[2, 2], padding=[[0, 0], [0, 0]], data_format=data_format, use_gpu=use_gpu)

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding1x1BackpropFilter(self):
        if False:
            return 10
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 2], output_sizes=[1, 3, 4, 2], strides=[1, 1], padding=[[1, 1], [1, 1]], data_format=data_format, use_gpu=use_gpu, err=5e-05)
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 2, 3, 2], filter_sizes=[1, 1, 2, 1], output_sizes=[1, 4, 3, 1], strides=[1, 2], padding=[[1, 1], [1, 1]], use_gpu=use_gpu, data_format=data_format)
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 4, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 4, 2, 1], strides=[1, 2], padding=[[1, 1], [1, 1]], data_format=data_format, use_gpu=use_gpu, dilations=[2, 2])

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding2x2BackpropFilter(self):
        if False:
            while True:
                i = 10
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[2, 3, 1, 1], filter_sizes=[2, 1, 1, 1], output_sizes=[2, 2, 5, 1], strides=[3, 1], padding=[[2, 2], [2, 2]], data_format=data_format, use_gpu=use_gpu)
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 3, 6, 1], filter_sizes=[3, 2, 1, 1], output_sizes=[1, 3, 4, 1], strides=[1, 2], padding=[[2, 2], [2, 2]], data_format=data_format, use_gpu=use_gpu, dilations=[2, 3])

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding_1_8_4_1_BackpropFilter(self):
        if False:
            print('Hello World!')
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 2, 3, 1], filter_sizes=[2, 2, 1, 1], output_sizes=[1, 10, 8, 1], strides=[1, 1], padding=[[1, 8], [4, 2]], data_format=data_format, use_gpu=use_gpu, err=0.0001)
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 5, 3, 1], filter_sizes=[3, 2, 1, 1], output_sizes=[1, 4, 8, 1], strides=[3, 1], padding=[[1, 8], [4, 2]], use_gpu=use_gpu, data_format=data_format)

    @test_util.run_in_graph_and_eager_modes()
    def testConv2D2x2Depth1Padding_5_0_2_2_BackpropFilter(self):
        if False:
            return 10
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 3, 3, 1], filter_sizes=[2, 1, 1, 1], output_sizes=[1, 7, 7, 1], strides=[1, 1], padding=[[5, 0], [2, 2]], data_format=data_format, use_gpu=use_gpu, err=0.0001)
            self._RunAndVerifyBackpropFilterExplicitPadding(input_sizes=[1, 4, 2, 1], filter_sizes=[3, 3, 1, 1], output_sizes=[1, 5, 2, 1], strides=[1, 2], padding=[[5, 0], [2, 2]], data_format=data_format, use_gpu=use_gpu, dilations=[2, 1])

    def ConstructAndTestGradient(self, batch, input_rows, input_cols, filter_rows, filter_cols, in_depth, out_depth, stride_rows, stride_cols, padding, test_input, data_format, use_gpu, num_groups=1, max_err=0.003):
        if False:
            return 10
        assert in_depth % num_groups == 0 and out_depth % num_groups == 0
        input_shape = [batch, input_rows, input_cols, in_depth]
        filter_shape = [filter_rows, filter_cols, in_depth // num_groups, out_depth]
        if padding == 'VALID':
            output_rows = (input_rows - filter_rows + stride_rows) // stride_rows
            output_cols = (input_cols - filter_cols + stride_cols) // stride_cols
        elif padding == 'SAME':
            output_rows = (input_rows + stride_rows - 1) // stride_rows
            output_cols = (input_cols + stride_cols - 1) // stride_cols
        else:
            self.assertIsInstance(padding, (list, tuple))
            output_rows = (input_rows + padding[1][0] + padding[1][1] - filter_rows + stride_rows) // stride_rows
            output_cols = (input_cols + padding[2][0] + padding[2][1] - filter_cols + stride_cols) // stride_cols
        output_shape = [batch, output_rows, output_cols, out_depth]
        input_size = 1
        for x in input_shape:
            input_size *= x
        filter_size = 1
        for x in filter_shape:
            filter_size *= x
        input_data = [x * 1.0 / input_size for x in range(0, input_size)]
        filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
        for dtype in self._DtypesToTest(use_gpu=use_gpu):
            with self.cached_session(use_gpu=use_gpu):
                input_tensor = constant_op.constant(input_data, shape=input_shape, dtype=dtype, name='input')
                filter_tensor = constant_op.constant(filter_data, shape=filter_shape, dtype=dtype, name='filter')
                strides = [1, stride_rows, stride_cols, 1]
                new_padding = padding
                if data_format == 'NCHW':
                    new_input_tensor = test_util.NHWCToNCHW(input_tensor)
                    strides = test_util.NHWCToNCHW(strides)
                    if isinstance(padding, (list, tuple)):
                        new_padding = test_util.NHWCToNCHW(padding)
                else:
                    new_input_tensor = input_tensor
                conv = nn_ops.conv2d(new_input_tensor, filter_tensor, strides, new_padding, data_format=data_format, name='conv')
                if data_format == 'NCHW':
                    conv = test_util.NCHWToNHWC(conv)
                self.assertEqual(output_shape, conv.get_shape())
                if test_input:
                    (jacob_t, jacob_n) = gradient_checker.compute_gradient(input_tensor, input_shape, conv, output_shape)
                else:
                    (jacob_t, jacob_n) = gradient_checker.compute_gradient(filter_tensor, filter_shape, conv, output_shape)
                if dtype == dtypes.float32:
                    reference_jacob_t = jacob_t
                    err = np.fabs(jacob_t - jacob_n).max()
                else:
                    err = np.fabs(jacob_t - reference_jacob_t).max()
                tf_logging.debug('conv_2d gradient error = %s', err)
                self.assertLess(err, max_err)

    @test_util.deprecated_graph_mode_only
    def testInputGradientValidPaddingStrideOne(self):
        if False:
            i = 10
            return i + 15
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=5, input_cols=4, filter_rows=3, filter_cols=3, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding='VALID', test_input=True, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testFilterGradientValidPaddingStrideOne(self):
        if False:
            return 10
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=4, input_rows=6, input_cols=5, filter_rows=2, filter_cols=2, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding='VALID', test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testInputGradientValidPaddingStrideTwo(self):
        if False:
            while True:
                i = 10
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=4, input_cols=5, filter_rows=3, filter_cols=3, in_depth=2, out_depth=3, stride_rows=2, stride_cols=2, padding='VALID', test_input=True, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testFilterGradientValidPaddingStrideTwo(self):
        if False:
            i = 10
            return i + 15
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=4, input_rows=6, input_cols=5, filter_rows=2, filter_cols=2, in_depth=2, out_depth=3, stride_rows=2, stride_cols=2, padding='VALID', test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testInputGradientValidPaddingStrideThree(self):
        if False:
            for i in range(10):
                print('nop')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=7, input_cols=6, filter_rows=3, filter_cols=3, in_depth=4, out_depth=5, stride_rows=3, stride_cols=3, padding='VALID', test_input=True, data_format=data_format, use_gpu=use_gpu, max_err=0.005)

    @test_util.deprecated_graph_mode_only
    def testFilterGradientValidPaddingStrideThree(self):
        if False:
            return 10
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=8, input_cols=7, filter_rows=4, filter_cols=4, in_depth=2, out_depth=3, stride_rows=3, stride_cols=3, padding='VALID', test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testInputGradientSamePaddingStrideOne(self):
        if False:
            i = 10
            return i + 15
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=7, input_cols=6, filter_rows=3, filter_cols=3, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding='SAME', test_input=True, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testFilterGradientSamePaddingStrideOne(self):
        if False:
            i = 10
            return i + 15
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=4, input_rows=6, input_cols=5, filter_rows=2, filter_cols=2, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding='SAME', test_input=False, data_format=data_format, use_gpu=use_gpu, max_err=0.005)

    @test_util.deprecated_graph_mode_only
    def testInputGradientSamePaddingStrideTwo(self):
        if False:
            for i in range(10):
                print('nop')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=5, input_cols=4, filter_rows=3, filter_cols=3, in_depth=3, out_depth=3, stride_rows=2, stride_cols=2, padding='SAME', test_input=True, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testFilterGradientSamePaddingStrideTwo(self):
        if False:
            for i in range(10):
                print('nop')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=4, input_rows=6, input_cols=5, filter_rows=2, filter_cols=2, in_depth=2, out_depth=3, stride_rows=2, stride_cols=2, padding='SAME', test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testInputGradientSamePaddingStrideThree(self):
        if False:
            print('Hello World!')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=7, input_cols=6, filter_rows=3, filter_cols=3, in_depth=4, out_depth=5, stride_rows=3, stride_cols=3, padding='SAME', test_input=True, data_format=data_format, use_gpu=use_gpu, max_err=0.005)

    @test_util.deprecated_graph_mode_only
    def testFilterGradientSamePaddingStrideThree(self):
        if False:
            print('Hello World!')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=8, input_cols=7, filter_rows=4, filter_cols=4, in_depth=2, out_depth=3, stride_rows=3, stride_cols=3, padding='SAME', test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testFilterGradientSamePaddingStride2x1(self):
        if False:
            i = 10
            return i + 15
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=8, input_cols=7, filter_rows=4, filter_cols=4, in_depth=2, out_depth=3, stride_rows=2, stride_cols=1, padding='SAME', test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testInputGradientKernelSizeMatchesInputSize(self):
        if False:
            for i in range(10):
                print('nop')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=4, input_cols=3, filter_rows=4, filter_cols=3, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding='VALID', test_input=True, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testFilterGradientKernelSizeMatchesInputSize(self):
        if False:
            i = 10
            return i + 15
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=4, input_cols=3, filter_rows=4, filter_cols=3, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding='VALID', test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testInputGradient1x1PaddingStrideOne(self):
        if False:
            i = 10
            return i + 15
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=5, input_cols=4, filter_rows=3, filter_cols=3, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], test_input=True, data_format=data_format, use_gpu=use_gpu, max_err=0.0025)

    @test_util.deprecated_graph_mode_only
    def testFilterGradient1x1PaddingStrideOne(self):
        if False:
            print('Hello World!')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=5, input_cols=4, filter_rows=3, filter_cols=3, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testInputGradient1x1PaddingStrideTwo(self):
        if False:
            return 10
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=4, input_cols=5, filter_rows=3, filter_cols=3, in_depth=2, out_depth=3, stride_rows=2, stride_cols=2, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], test_input=True, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testFilterGradient1x1PaddingStrideTwo(self):
        if False:
            print('Hello World!')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=4, input_cols=5, filter_rows=3, filter_cols=3, in_depth=2, out_depth=3, stride_rows=2, stride_cols=2, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testInputGradient2x2PaddingStrideOne(self):
        if False:
            while True:
                i = 10
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=5, input_cols=4, filter_rows=3, filter_cols=3, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], test_input=True, data_format=data_format, use_gpu=use_gpu, max_err=0.003)

    @test_util.deprecated_graph_mode_only
    def testFilterGradient2x2PaddingStrideOne(self):
        if False:
            return 10
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=5, input_cols=4, filter_rows=3, filter_cols=3, in_depth=2, out_depth=3, stride_rows=1, stride_cols=1, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], test_input=False, data_format=data_format, use_gpu=use_gpu, max_err=0.005)

    @test_util.deprecated_graph_mode_only
    def testInputGradient1_2_3_4PaddingStride3x2(self):
        if False:
            print('Hello World!')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=8, input_cols=5, filter_rows=4, filter_cols=2, in_depth=3, out_depth=2, stride_rows=3, stride_cols=2, padding=[[0, 0], [1, 2], [3, 4], [0, 0]], test_input=True, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testFilterGradient1_2_3_4PaddingStride3x2(self):
        if False:
            print('Hello World!')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=8, input_cols=5, filter_rows=4, filter_cols=2, in_depth=3, out_depth=2, stride_rows=3, stride_cols=2, padding=[[0, 0], [1, 2], [3, 4], [0, 0]], test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testInputGradient4_3_2_1PaddingStride2x1(self):
        if False:
            print('Hello World!')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=3, input_rows=5, input_cols=7, filter_rows=3, filter_cols=2, in_depth=1, out_depth=2, stride_rows=2, stride_cols=1, padding=[[0, 0], [4, 3], [2, 1], [0, 0]], test_input=True, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testFilterGradient4_3_2_1PaddingStride2x1(self):
        if False:
            return 10
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=3, input_rows=5, input_cols=7, filter_rows=3, filter_cols=2, in_depth=1, out_depth=2, stride_rows=2, stride_cols=1, padding=[[0, 0], [4, 3], [2, 1], [0, 0]], test_input=False, data_format=data_format, use_gpu=use_gpu, max_err=0.005)

    @test_util.deprecated_graph_mode_only
    def testInputGradient0_0_0_5PaddingStride1x2(self):
        if False:
            for i in range(10):
                print('nop')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=6, input_cols=7, filter_rows=3, filter_cols=4, in_depth=3, out_depth=2, stride_rows=1, stride_cols=2, padding=[[0, 0], [0, 0], [0, 5], [0, 0]], test_input=True, data_format=data_format, use_gpu=use_gpu, max_err=0.005)

    @test_util.deprecated_graph_mode_only
    def testFilterGradient0_0_0_5PaddingStride1x2(self):
        if False:
            for i in range(10):
                print('nop')
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(batch=2, input_rows=6, input_cols=7, filter_rows=3, filter_cols=4, in_depth=3, out_depth=2, stride_rows=1, stride_cols=2, padding=[[0, 0], [0, 0], [0, 5], [0, 0]], test_input=False, data_format=data_format, use_gpu=use_gpu)

    @test_util.deprecated_graph_mode_only
    def testShapeFunctionEdgeCases(self):
        if False:
            for i in range(10):
                print('nop')
        c1 = nn_ops.conv2d(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.float32), strides=[1, 1, 1, 1], padding='SAME')
        self.assertEqual([None, None, None, None], c1.get_shape().as_list())
        with self.assertRaises(ValueError):
            nn_ops.conv2d(array_ops.placeholder(dtypes.float32, shape=[1, 3]), array_ops.placeholder(dtypes.float32), strides=[1, 1, 1, 1], padding='SAME')
        with self.assertRaises(ValueError):
            nn_ops.conv2d(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.float32, shape=[1, 3]), strides=[1, 1, 1, 1], padding='SAME')
        with self.assertRaises(ValueError):
            nn_ops.conv2d(array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]), array_ops.placeholder(dtypes.float32, shape=[4, 4, 2, 2]), strides=[1, 1, 1, 1], padding='SAME')
        nn_ops.conv2d(array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 8]), array_ops.placeholder(dtypes.float32, shape=[4, 4, 2, 16]), strides=[1, 1, 1, 1], padding='SAME')
        with self.assertRaises(ValueError):
            nn_ops.conv2d(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.float32), strides=[1, 1, 1, 1], padding=[[0, 0], [0, -1], [1, 2], [0, 0]])
        with self.assertRaises(ValueError):
            nn_ops.conv2d(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.float32), strides=[1, 1, 1, 1], padding=[[1, 0], [0, 0], [0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            nn_ops.conv2d(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.float32), strides=[1, 1, 1, 1], padding=[[0, 0], [0, 1], [0, 0], [0, 0]], data_format='NCHW')
        with self.assertRaises(ValueError):
            nn_ops.conv2d(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.float32), strides=[1, 1, 1, 1], padding=[[0, 0], [0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            nn_ops.conv2d(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.float32), strides=[1, 1, 1, 1], padding=[[0], [0], [0], [0]])
        with self.assertRaises(ValueError):
            nn_ops.conv2d(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.float32), strides=[1, 1, 1, 1], padding=[0, 0, 0, 0])

    def testOpEdgeCases(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex((ValueError, errors_impl.UnimplementedError), 'strides in the batch and depth'):
            input_val = np.ones([2, 4, 10, 10])
            filter_val = np.ones([2, 4, 10, 10])
            self.evaluate(nn_ops.conv2d(input_val, filter_val, strides=[2, 1, 1, 1], padding='SAME'))
        with self.assertRaisesRegex((ValueError, errors_impl.UnimplementedError), 'strides in the batch and depth'):
            input_val = np.ones([2, 4, 10, 10])
            filter_val = np.ones([2, 4, 10, 10])
            self.evaluate(nn_ops.conv2d(input_val, filter_val, strides=[1, 1, 1, 2], padding='SAME'))
        with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'filter must not have zero elements|has a non-positive dimension'):
            input_val = np.ones([1, 1, 1, 1])
            filter_val = np.ones([1, 0, 1, 1])
            self.evaluate(nn_ops.conv2d(input_val, filter_val, strides=[1, 1, 1, 1], padding='SAME'))
        with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'All elements of explicit_paddings must be nonnegative'):
            filter_val = np.ones([18, 18, 3, 2])
            out_backprop_val = np.ones([32, 3, 2, 2])
            self.evaluate(nn_ops.conv2d_backprop_input([32, 20, 20, 3], filter_val, out_backprop_val, strides=[1, 1, 1, 1], padding=[[0, 0], [-1, 0], [0, 0], [0, 0]]))
        with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'All elements of explicit_paddings must be nonnegative'):
            input_val = np.ones([32, 20, 20, 3])
            out_backprop_val = np.ones([32, 3, 2, 2])
            self.evaluate(nn_ops.conv2d_backprop_filter(input_val, [18, 18, 3, 2], out_backprop_val, strides=[1, 1, 1, 1], padding=[[0, 0], [-1, 0], [0, 0], [0, 0]]))

    def testConvOpEdgeCases(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex((errors_impl.InvalidArgumentError, errors_impl.UnimplementedError), 'strides in the batch and depth'):
            input_val = np.ones([2, 4, 10, 10])
            filter_val = np.ones([2, 4, 10, 10])
            self.evaluate(gen_nn_ops.conv(input_val, filter_val, strides=[2, 1, 1, 1], padding='SAME'))
        with self.assertRaisesRegex((errors_impl.InvalidArgumentError, errors_impl.UnimplementedError), 'strides in the batch and depth'):
            input_val = np.ones([2, 4, 10, 10])
            filter_val = np.ones([2, 4, 10, 10])
            self.evaluate(gen_nn_ops.conv(input_val, filter_val, strides=[1, 1, 1, 2], padding='SAME'))
        with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'filter must not have zero elements|has a non-positive dimension'):
            input_val = np.ones([1, 1, 1, 1])
            filter_val = np.ones([1, 0, 1, 1])
            self.evaluate(gen_nn_ops.conv(input_val, filter_val, strides=[1, 1, 1, 1], padding='SAME'))

    def testConv2DBackpropInputInvalidOutBackpropRaiseError(self):
        if False:
            while True:
                i = 10
        with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
            with self.cached_session():
                input_sizes = constant_op.constant([65534, 65534], shape=[2], dtype=dtypes.int32)
                filters = constant_op.constant(0.159749106, shape=[3, 3, 2, 2], dtype=dtypes.float32)
                out_backprop = constant_op.constant(0, shape=[], dtype=dtypes.float32)
                t = gen_nn_ops.conv2d_backprop_input(input_sizes=input_sizes, filter=filters, out_backprop=out_backprop, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True, explicit_paddings=[], data_format='NHWC', dilations=[1, 1, 1, 1])
                self.evaluate(t)

@test_util.run_all_without_tensor_float_32('Avoid TF32 conv on GPU')
class DepthwiseConv2DTest(test.TestCase):

    def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding, expected):
        if False:
            print('Hello World!')
        'Verifies the output values of the convolution function.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,\n        input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in [filter_rows, filter_cols,\n        input_depth, depth_multiplier].\n      stride: Stride.\n      padding: Padding type.\n      expected: An array containing the expected operation outputs.\n    '
        total_size_1 = 1
        total_size_2 = 1
        for s in tensor_in_sizes:
            total_size_1 *= s
        for s in filter_in_sizes:
            total_size_2 *= s
        x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
        x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
        with self.cached_session():
            t1 = constant_op.constant(x1, shape=tensor_in_sizes)
            t1.set_shape(tensor_in_sizes)
            t2 = constant_op.constant(x2, shape=filter_in_sizes)
            conv = nn_impl.depthwise_conv2d(t1, t2, strides=[1, stride, stride, 1], padding=padding)
            value = self.evaluate(conv)
        tf_logging.debug('value = %s', value)
        self.assertArrayNear(expected, np.ravel(value), 1e-05)
        self.assertShapeEqual(value, conv)

    def testConv2D2x2Filter(self):
        if False:
            i = 10
            return i + 15
        expected_output = [196, 216, 272, 296, 252, 280, 344, 376]
        self._VerifyValues(tensor_in_sizes=[1, 2, 3, 2], filter_in_sizes=[2, 2, 2, 2], stride=1, padding='VALID', expected=expected_output)

@test_util.run_all_without_tensor_float_32('Avoid TF32 conv on GPU')
class SeparableConv2DTest(test.TestCase):

    def _InitValues(self, sizes):
        if False:
            while True:
                i = 10
        'Initializes values for input tensors.\n\n    Args:\n      sizes: Tensor dimensions.\n\n    Returns:\n      Tensor initialized to values.\n    '
        total_size = 1
        for s in sizes:
            total_size *= s
        x = [f * 0.5 for f in range(1, total_size + 1)]
        return constant_op.constant(x, shape=sizes)

    def _VerifyValues(self, tensor_in_sizes, depthwise_filter_in_sizes, pointwise_filter_in_sizes, stride, padding, expected, data_format='NHWC'):
        if False:
            for i in range(10):
                print('nop')
        'Verifies the output values of the separable convolution function.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions.\n      depthwise_filter_in_sizes: Depthwise filter tensor dimensions.\n      pointwise_filter_in_sizes: Pointwise filter tensor dimensions.\n      stride: Stride.\n      padding: Padding type.\n      expected: An array containing the expected operation outputs.\n      data_format: string data format for input tensor.\n    '
        with self.cached_session():
            t1 = self._InitValues(tensor_in_sizes)
            f1 = self._InitValues(depthwise_filter_in_sizes)
            f1.set_shape(depthwise_filter_in_sizes)
            f2 = self._InitValues(pointwise_filter_in_sizes)
            real_t1 = t1
            strides = [1, stride, stride, 1]
            if data_format == 'NCHW':
                real_t1 = array_ops.transpose(t1, [0, 3, 1, 2])
                strides = [1, 1, stride, stride]
                if isinstance(padding, list):
                    padding = [padding[0], padding[3], padding[1], padding[2]]
            conv = nn_impl.separable_conv2d(real_t1, f1, f2, strides=strides, padding=padding, data_format=data_format)
            if data_format == 'NCHW':
                conv = array_ops.transpose(conv, [0, 2, 3, 1])
            value = self.evaluate(conv)
        tf_logging.debug('value = %s', value)
        self.assertArrayNear(expected, np.ravel(value), 0.002)
        self.assertShapeEqual(value, conv)

    def _testSeparableConv2D(self, data_format):
        if False:
            i = 10
            return i + 15
        expected_output = [6644.5, 6971.5, 7298.5, 7625.5, 7952.5, 8279.5, 8606.5, 8154.5, 8556.5, 8958.5, 9360.5, 9762.5, 10164.5, 10566.5, 9664.5, 10141.5, 10618.5, 11095.5, 11572.5, 12049.5, 12526.5, 4145.5, 4346.5, 4547.5, 4748.5, 4949.5, 5150.5, 5351.5, 12684.5, 13311.5, 13938.5, 14565.5, 15192.5, 15819.5, 16446.5, 14194.5, 14896.5, 15598.5, 16300.5, 17002.5, 17704.5, 18406.5, 15704.5, 16481.5, 17258.5, 18035.5, 18812.5, 19589.5, 20366.5, 6499.5, 6814.5, 7129.5, 7444.5, 7759.5, 8074.5, 8389.5, 18724.5, 19651.5, 20578.5, 21505.5, 22432.5, 23359.5, 24286.5, 20234.5, 21236.5, 22238.5, 23240.5, 24242.5, 25244.5, 26246.5, 21744.5, 22821.5, 23898.5, 24975.5, 26052.5, 27129.5, 28206.5, 8853.5, 9282.5, 9711.5, 10140.5, 10569.5, 10998.5, 11427.5, 5746.75, 6010.75, 6274.75, 6538.75, 6802.75, 7066.75, 7330.75, 6168.75, 6452.25, 6735.75, 7019.25, 7302.75, 7586.25, 7869.75, 6590.75, 6893.75, 7196.75, 7499.75, 7802.75, 8105.75, 8408.75, 2036.25, 2119.5, 2202.75, 2286.0, 2369.25, 2452.5, 2535.75]
        self._VerifyValues(tensor_in_sizes=[1, 4, 4, 2], depthwise_filter_in_sizes=[2, 2, 2, 3], pointwise_filter_in_sizes=[1, 1, 6, 7], stride=1, padding='SAME', expected=expected_output, data_format=data_format)

    def testSeparableConv2D(self):
        if False:
            while True:
                i = 10
        self._testSeparableConv2D('NHWC')

    def disabledtestSeparableConv2DNCHW(self):
        if False:
            while True:
                i = 10
        if not test.is_gpu_available():
            return
        self._testSeparableConv2D('NCHW')

    def _testSeparableConv2DEqualInputOutputDepth(self, data_format):
        if False:
            i = 10
            return i + 15
        expected_output = [5742.0, 6069.0, 6396.0, 6723.0, 7050.0, 7377.0, 7047.0, 7449.0, 7851.0, 8253.0, 8655.0, 9057.0, 8352.0, 8829.0, 9306.0, 9783.0, 10260.0, 10737.0, 3582.0, 3783.0, 3984.0, 4185.0, 4386.0, 4587.0, 10962.0, 11589.0, 12216.0, 12843.0, 13470.0, 14097.0, 12267.0, 12969.0, 13671.0, 14373.0, 15075.0, 15777.0, 13572.0, 14349.0, 15126.0, 15903.0, 16680.0, 17457.0, 5616.0, 5931.0, 6246.0, 6561.0, 6876.0, 7191.0, 16182.0, 17109.0, 18036.0, 18963.0, 19890.0, 20817.0, 17487.0, 18489.0, 19491.0, 20493.0, 21495.0, 22497.0, 18792.0, 19869.0, 20946.0, 22023.0, 23100.0, 24177.0, 7650.0, 8079.0, 8508.0, 8937.0, 9366.0, 9795.0, 4963.5, 5227.5, 5491.5, 5755.5, 6019.5, 6283.5, 5328.0, 5611.5, 5895.0, 6178.5, 6462.0, 6745.5, 5692.5, 5995.5, 6298.5, 6601.5, 6904.5, 7207.5, 1757.25, 1840.5, 1923.75, 2007.0, 2090.25, 2173.5]
        self._VerifyValues(tensor_in_sizes=[1, 4, 4, 2], depthwise_filter_in_sizes=[2, 2, 2, 3], pointwise_filter_in_sizes=[1, 1, 6, 6], stride=1, padding='SAME', expected=expected_output, data_format=data_format)

    @test_util.deprecated_graph_mode_only
    def testSeparableConv2DEqualInputOutputDepth(self):
        if False:
            for i in range(10):
                print('nop')
        self._testSeparableConv2DEqualInputOutputDepth('NHWC')

    def testSeparableConv2DEqualInputOutputDepthNCHW(self):
        if False:
            print('Hello World!')
        if not test.is_gpu_available():
            return
        self._testSeparableConv2DEqualInputOutputDepth('NCHW')

    def _testSeparableConv2dExplicitPadding(self, data_format):
        if False:
            return 10
        tensor_in_sizes = [1, 4, 4, 2]
        depthwise_filter_in_sizes = [2, 2, 2, 3]
        pointwise_filter_in_sizes = [1, 1, 6, 7]
        padding = [[0, 0], [1, 2], [3, 4], [0, 0]]
        with self.cached_session():
            t1 = self._InitValues(tensor_in_sizes)
            t1 = array_ops.pad(t1, padding)
            f1 = self._InitValues(depthwise_filter_in_sizes)
            f1.set_shape(depthwise_filter_in_sizes)
            f2 = self._InitValues(pointwise_filter_in_sizes)
            conv = nn_impl.separable_conv2d(t1, f1, f2, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
            expected = self.evaluate(conv)
            expected = np.ravel(expected)
        self._VerifyValues(tensor_in_sizes=tensor_in_sizes, depthwise_filter_in_sizes=depthwise_filter_in_sizes, pointwise_filter_in_sizes=pointwise_filter_in_sizes, stride=1, padding=padding, expected=expected, data_format=data_format)

    def testSeparableConv2dExplicitPadding(self):
        if False:
            return 10
        self._testSeparableConv2dExplicitPadding('NHWC')

    def testSeparableConv2dExplicitPaddingNCHW(self):
        if False:
            return 10
        if not test.is_gpu_available():
            return
        self._testSeparableConv2dExplicitPadding('NCHW')

@test_util.run_all_without_tensor_float_32('Avoid TF32 conv on GPU')
class DeepConv2DTest(test.TestCase):

    def _CompareFwdConv2D(self, tensor_in_sizes, filter_in_sizes, conv_strides, padding):
        if False:
            while True:
                i = 10
        'Verifies that DeepConv2D and Conv2D produce the same values.\n\n    Args:\n      tensor_in_sizes: Input tensor dimensions in\n        [batch, input_rows, input_cols, input_depth].\n      filter_in_sizes: Filter tensor dimensions in\n        [kernel_rows, kernel_cols, input_depth, output_depth].\n      conv_strides: [row_stride, col_stride] for the convolution;\n      padding: Padding type.\n    '
        x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
        x2 = np.random.rand(*filter_in_sizes).astype(np.float32)
        with self.cached_session(use_gpu=False):
            t1 = constant_op.constant(x1, shape=tensor_in_sizes)
            t2 = constant_op.constant(x2, shape=filter_in_sizes)
            strides = [1] + conv_strides + [1]
            conv = nn_ops.conv2d(t1, t2, strides=strides, padding=padding)
            os.environ['TF_USE_DEEP_CONV2D'] = '0'
            values_expect = self.evaluate([conv])
            os.environ['TF_USE_DEEP_CONV2D'] = '1'
            values_test = self.evaluate([conv])
            self.assertAllClose(values_expect, values_test, rtol=1e-05, atol=1e-05)

    def _RunTestCases(self, conv_strides, padding):
        if False:
            for i in range(10):
                print('nop')
        input_sizes = [[5, 5, 5, 1248], [3, 17, 17, 192], [2, 35, 35, 288], [2, 6, 8, 517], [2, 7, 4, 81], [3, 11, 3, 77]]
        filter_sizes = [[3, 3, 1248, 128], [3, 3, 192, 192], [3, 3, 288, 384], [3, 3, 517, 64], [3, 3, 81, 77], [3, 3, 77, 181]]
        for (input_shape, filter_shape) in zip(input_sizes, filter_sizes):
            self._CompareFwdConv2D(input_shape, filter_shape, conv_strides, padding)

    def testConv2D3x3FilterStride1x1Valid(self):
        if False:
            print('Hello World!')
        self._RunTestCases([1, 1], 'VALID')

    def testConv2D3x3FilterStride1x1Same(self):
        if False:
            print('Hello World!')
        self._RunTestCases([1, 1], 'SAME')

class Conv2DBenchmark(test.Benchmark):

    def benchmarkGPUConvStackFirst(self):
        if False:
            while True:
                i = 10
        if not test.is_gpu_available():
            return
        with ops.Graph().as_default(), session_lib.Session() as session:
            batch_size = 1
            timesteps = 600
            features = 1
            inputs = random_ops.random_uniform([batch_size, 1, timesteps, features], seed=1234)
            num_outputs_list = [512] * 40 + [1]
            kernel_w = 3
            x = inputs
            for num_outputs in num_outputs_list:
                x = convolutional.conv2d(x, num_outputs, [1, kernel_w])
            outputs = x
            self.evaluate(variables.global_variables_initializer())
            num_iterations = 4
            for iter_index in range(num_iterations):
                start = time.time()
                session.run(outputs)
                wall_time = time.time() - start
                self.report_benchmark(name='conv_stack_iter_%d' % iter_index, wall_time=wall_time)
                tf_logging.info('conv_stack_iter_%d: %.4f' % (iter_index, wall_time))

    def _bench_op(self, name, op, burn_iters, num_iters):
        if False:
            while True:
                i = 10
        config = config_pb2.ConfigProto()
        config.graph_options.rewrite_options.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
        with session_lib.Session(config=config) as session:
            self.evaluate(variables.global_variables_initializer())
            self.run_op_benchmark(session, op, burn_iters=burn_iters, min_iters=num_iters, name=name)

    def benchmarkExplicitVsManualPadding(self):
        if False:
            i = 10
            return i + 15
        'Compare performance of EXPLICIT padding and calling tf.pad.\n\n    A Conv2D op with EXPLICIT padding is benchmarked, and a tf.pad with the same\n    padding followed by an equivalent Conv2D op is benchmarked.\n    '
        if not test.is_gpu_available():
            return
        with ops.Graph().as_default():
            burn_iters = 15
            num_iters = 300
            batch_size = 64
            input = variables.Variable(random_ops.random_uniform([batch_size, 3, 224, 224]))
            filter = variables.Variable(random_ops.random_uniform([7, 7, 3, 64]))
            strides = [1, 1, 2, 2]
            padding = [(0, 0), (0, 0), (3, 3), (3, 3)]
            output_explicit_pad = nn_ops.conv2d(input, filter, strides, padding=padding, data_format='NCHW')
            input_padded = array_ops.pad(input, padding)
            output_manual_pad = nn_ops.conv2d(input_padded, filter, strides, padding='VALID', data_format='NCHW')
            self._bench_op('explicit_pad_forward', output_explicit_pad.op, burn_iters, num_iters)
            self._bench_op('manual_pad_forward', output_manual_pad.op, burn_iters, num_iters)
            (input_grad_explicit_pad, filter_grad_explicit_pad) = gradients_impl.gradients(output_explicit_pad, [input, filter])
            self._bench_op('explicit_pad_backward', control_flow_ops.group(input_grad_explicit_pad, filter_grad_explicit_pad), burn_iters, num_iters)
            (input_grad_manual_pad, filter_grad_manual_pad) = gradients_impl.gradients(output_manual_pad, [input, filter])
            self._bench_op('manual_pad_backward', control_flow_ops.group(input_grad_manual_pad, filter_grad_manual_pad), burn_iters, num_iters)

    def benchmarkExplicitVsSamePaddingGraph(self):
        if False:
            i = 10
            return i + 15
        'Compare performance of EXPLICIT and SAME padding in graph mode.\n\n    A Conv2D op with SAME padding is benchmarked, and an equivalent Conv2D op\n    with explicit padding is benchmarked, where the padding is the same as in\n    the SAME case. The purpose is to ensure EXPLICIT padding is just as\n    efficient as the SAME case\n    '
        if not test.is_gpu_available():
            return
        with ops.Graph().as_default():
            burn_iters = 15
            num_convs = 20
            num_iters = 50
            batch_size = 64
            input = variables.Variable(random_ops.random_uniform([batch_size, 256, 14, 14]))
            filter = variables.Variable(random_ops.random_uniform([3, 3, 256, 256]))
            strides = [1, 1, 1, 1]
            padding = [(0, 0), (0, 0), (1, 1), (1, 1)]
            output_explicit_pad = input
            output_same_pad = input
            for _ in range(num_convs):
                output_explicit_pad = nn_ops.conv2d(output_explicit_pad, filter, strides, padding=padding, data_format='NCHW')
                output_same_pad = nn_ops.conv2d(output_same_pad, filter, strides, padding='SAME', data_format='NCHW')
            (grad_explicit_pad,) = gradients_impl.gradients(output_explicit_pad, filter)
            (grad_same_pad,) = gradients_impl.gradients(output_same_pad, filter)
            self._bench_op('graph_explicit_pad', grad_explicit_pad.op, burn_iters, num_iters)
            self._bench_op('graph_same_pad', grad_same_pad.op, burn_iters, num_iters)

    def benchmarkExplicitVsSamePaddingEager(self):
        if False:
            while True:
                i = 10
        'Compare performance of EXPLICIT and SAME padding in eager mode.\n\n    A Conv2D op with SAME padding is benchmarked, and an equivalent Conv2D op\n    with explicit padding is benchmarked, where the padding is the same as in\n    the SAME case. Currently, EXPLICIT padding is slightly slower, due to the\n    fact the Python padding list must be checked and processed before the Conv2D\n    op can run.\n    '
        if not test.is_gpu_available():
            return
        with context.eager_mode():
            burn_iters = 15
            num_convs = 20
            num_iters = 50
            batch_size = 64
            input = variables.Variable(random_ops.random_uniform([batch_size, 256, 14, 14]))
            filter = variables.Variable(random_ops.random_uniform([3, 3, 256, 256]))
            strides = [1, 1, 1, 1]
            padding = [(0, 0), (0, 0), (1, 1), (1, 1)]
            output_explicit_pad = input
            output_same_pad = input
            for _ in range(burn_iters):
                output_explicit_pad = nn_ops.conv2d(output_explicit_pad, filter, strides, padding=padding, data_format='NCHW')
                output_same_pad = nn_ops.conv2d(output_same_pad, filter, strides, padding='SAME', data_format='NCHW')
            start = time.time()
            for _ in range(num_iters):
                with backprop.GradientTape() as tape:
                    for _ in range(num_convs):
                        output_explicit_pad = nn_ops.conv2d(output_explicit_pad, filter, strides, padding=padding, data_format='NCHW')
                    tape.gradient(output_explicit_pad, filter)
            end = time.time()
            self.report_benchmark(name='eager_explicit_pad', wall_time=(end - start) / num_iters, iters=num_iters)
            start = time.time()
            for _ in range(num_iters):
                with backprop.GradientTape() as tape:
                    for _ in range(num_convs):
                        output_same_pad = nn_ops.conv2d(output_same_pad, filter, strides, padding='SAME', data_format='NCHW')
                    tape.gradient(output_same_pad, filter)
            end = time.time()
            self.report_benchmark(name='eager_same_pad', wall_time=(end - start) / num_iters, iters=num_iters)

def GetInceptionFwdTest(input_size, filter_size, stride, padding, gpu_only=False):
    if False:
        i = 10
        return i + 15

    def Test(self):
        if False:
            for i in range(10):
                print('nop')
        if gpu_only and (not test.is_gpu_available()):
            tf_logging.info('Skipping InceptionFwd %s', (input_size, filter_size, stride, padding))
            return
        tf_logging.info('Testing InceptionFwd %s', (input_size, filter_size, stride, padding))
        self._CompareFwdValues(input_size, filter_size, [stride, stride], padding)
    return Test

def GetInceptionFwdDilatedConvTest(input_size, filter_size, stride, padding):
    if False:
        i = 10
        return i + 15

    def Test(self):
        if False:
            while True:
                i = 10
        if stride == 1:
            tf_logging.info('Testing InceptionFwd with dilations %s', (input_size, filter_size, stride, padding))
            self._VerifyDilatedConvValues(tensor_in_sizes=input_size, filter_in_sizes=filter_size, strides=[stride, stride], dilations=[2, 2], padding=padding, rtol=0.0005)
    return Test

def GetInceptionBackInputTest(input_size, filter_size, output_size, stride, padding, gpu_only=False):
    if False:
        while True:
            i = 10

    def Test(self):
        if False:
            print('Hello World!')
        if gpu_only and (not test.is_gpu_available()):
            tf_logging.info('Skipping InceptionBackInput %s', (input_size, filter_size, output_size, stride, padding))
            return
        tf_logging.info('Testing InceptionBackInput %s', (input_size, filter_size, output_size, stride, padding))
        self._CompareBackpropInput(input_size, filter_size, output_size, [stride, stride], padding)
    return Test

def GetInceptionBackFilterTest(input_size, filter_size, output_size, strides, padding, gpu_only=False):
    if False:
        return 10

    def Test(self):
        if False:
            i = 10
            return i + 15
        if gpu_only and (not test.is_gpu_available()):
            tf_logging.info('Skipping InceptionBackFilter %s', (input_size, filter_size, output_size, strides, padding))
            return
        tf_logging.info('Testing InceptionBackFilter %s', (input_size, filter_size, output_size, strides, padding))
        self._CompareBackFilter(input_size, filter_size, output_size, strides, padding)
    return Test

@test_util.run_all_without_tensor_float_32('Avoid TF32 conv on GPU')
class FusedConv2DTest(test.TestCase):

    def _CreateNumpyTensor(self, shape):
        if False:
            return 10
        total_size = np.prod(shape)
        return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

    def _CreateConv2D(self, input_values, filters, strides=[1, 1], padding='SAME'):
        if False:
            i = 10
            return i + 15
        return nn_ops.convolution(input_values, filters, strides=strides, padding=padding)

    @test_util.run_in_graph_and_eager_modes(use_gpu=False)
    def testAddWithRefCountOne(self):
        if False:
            for i in range(10):
                print('nop')
        expected_output = [113377, 125570, 77305, 86738, 19433, 22226, 60681, 70722, 36291, 43718, 7143, 9206, 9785, 12098, 4783, 6366, 779, 1134]
        tensor_in_sizes = [1, 3, 3, 2]
        filter_in_sizes = [2, 2, 2, 2]
        bias_in_sizes = [2]
        x = self._CreateNumpyTensor(tensor_in_sizes)
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        bias_in = self._CreateNumpyTensor(bias_in_sizes)
        offset = 1
        conv1 = self._CreateConv2D(x, filter_in)
        conv2 = self._CreateConv2D(conv1, filter_in + offset)
        conv = self._CreateConv2D(conv1, filter_in - offset)
        bias_add = nn_ops.bias_add(conv, bias_in)
        add = math_ops.add_n([bias_add, conv2])
        self.assertAllEqual(np.rint(expected_output), self.evaluate(add).reshape(-1))

    @test_util.run_in_graph_and_eager_modes(use_gpu=False)
    def testAddWithRefCountTwoAndRunAddLast(self):
        if False:
            return 10
        expected_output = [1907175.0, 2253505.0, 780921.0, 953718.0, 118417.0, 152307.0, 536701.0, 680370.0, 186709.0, 252946.0, 23623.0, 35226.0, 51217.0, 71683.0, 14943.0, 23474.0, 1558.0, 2903.0]
        tensor_in_sizes = [1, 3, 3, 2]
        filter_in_sizes = [2, 2, 2, 2]
        bias_in_sizes = [2]
        x = self._CreateNumpyTensor(tensor_in_sizes)
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        bias_in = self._CreateNumpyTensor(bias_in_sizes)
        offset = 1
        conv1 = self._CreateConv2D(x, filter_in)
        conv2 = self._CreateConv2D(conv1, filter_in + offset)
        conv = self._CreateConv2D(conv2, filter_in - offset)
        bias_add = nn_ops.bias_add(conv, bias_in)
        add = math_ops.add_n([bias_add, conv1])
        self.assertAllEqual(np.rint(expected_output), self.evaluate(add).reshape(-1))

    @test_util.run_in_graph_and_eager_modes(use_gpu=False)
    def testAddWithRefCountTwoAndRunAddFirst(self):
        if False:
            print('Hello World!')
        expected_output = [176161, 194450, 120673, 134822, 30545, 34734, 96041, 111102, 58149, 69289, 11745, 14839, 15833, 19302, 7965, 10339, 1345, 1877]
        tensor_in_sizes = [1, 3, 3, 2]
        filter_in_sizes = [2, 2, 2, 2]
        bias_in_sizes = [2]
        x = self._CreateNumpyTensor(tensor_in_sizes)
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        bias_in = self._CreateNumpyTensor(bias_in_sizes)
        offset = 1
        conv1 = self._CreateConv2D(x, filter_in)
        conv2 = self._CreateConv2D(conv1, filter_in + offset)
        conv = self._CreateConv2D(conv1, filter_in - offset)
        bias_add = nn_ops.bias_add(conv, bias_in)
        add = math_ops.add_n([bias_add, conv2])
        relu = nn_ops.relu(add)
        output = math_ops.add_n([relu, conv2])
        self.assertAllEqual(np.rint(expected_output), self.evaluate(output).reshape(-1))

    @test_util.run_in_graph_and_eager_modes(use_gpu=False)
    def testAddWithRefCountTwoAndNoDependence(self):
        if False:
            i = 10
            return i + 15
        expected_output = [176161, 194450, 120673, 134822, 30545, 34734, 96041, 111102, 58149, 69289, 11745, 14839, 15833, 19302, 7965, 10339, 1345, 1877]
        tensor_in_sizes = [1, 3, 3, 2]
        filter_in_sizes = [2, 2, 2, 2]
        bias_in_sizes = [2]
        x = self._CreateNumpyTensor(tensor_in_sizes)
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        bias_in = self._CreateNumpyTensor(bias_in_sizes)
        offset = 1
        conv1 = self._CreateConv2D(x, filter_in)
        conv2 = self._CreateConv2D(conv1, filter_in + offset)
        conv = self._CreateConv2D(conv1, filter_in - offset)
        bias_add = nn_ops.bias_add(conv, bias_in)
        add = math_ops.add_n([bias_add, conv2])
        relu1 = nn_ops.relu(add)
        relu2 = nn_ops.relu(conv2)
        output = math_ops.add_n([relu1, relu2])
        self.assertAllEqual(np.rint(expected_output), self.evaluate(output).reshape(-1))

    @test_util.run_in_graph_and_eager_modes(use_gpu=False)
    def testAddWithSameSrcAndAddTensorBuffer(self):
        if False:
            i = 10
            return i + 15
        expected_output = [57157, 63298, 39249, 44026, 9971, 11402, 31193, 36306, 19126, 22948, 3970, 5060, 5135, 6350, 2666, 3524, 461, 674]
        tensor_in_sizes = [1, 3, 3, 2]
        filter_in_sizes = [2, 2, 2, 2]
        bias_in_sizes = [2]
        x = self._CreateNumpyTensor(tensor_in_sizes)
        filter_in = self._CreateNumpyTensor(filter_in_sizes)
        bias_in = self._CreateNumpyTensor(bias_in_sizes)
        conv1 = self._CreateConv2D(x, filter_in)
        conv = self._CreateConv2D(conv1, filter_in)
        bias_add = nn_ops.bias_add(conv, bias_in)
        add = math_ops.add_n([bias_add, conv1])
        self.assertAllEqual(np.rint(expected_output), self.evaluate(add).reshape(-1))

    @test_util.run_in_graph_and_eager_modes()
    def testResizeAndPadLargeResize(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError), 'Encountered overflow'):
            mode = 'REFLECT'
            strides = [1, 1, 1, 1]
            padding = 'SAME'
            resize_align_corners = False
            tensor = constant_op.constant(147, shape=[3, 3, 1, 4], dtype=dtypes.float32)
            size = constant_op.constant([1879048192, 1879048192], dtype=dtypes.int32)
            paddings = constant_op.constant([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=dtypes.int32)
            kernel = constant_op.constant(123, shape=[1, 3, 4, 1], dtype=dtypes.float32)
            self.evaluate(gen_nn_ops.fused_resize_and_pad_conv2d(input=tensor, size=size, paddings=paddings, filter=kernel, mode=mode, strides=strides, padding=padding, resize_align_corners=resize_align_corners))
if __name__ == '__main__':
    for (index, (input_size_, filter_size_, output_size_, stride_, padding_)) in enumerate(GetShrunkInceptionShapes()):
        setattr(Conv2DTest, 'testInceptionFwd_' + str(index), test_util.run_in_graph_and_eager_modes(GetInceptionFwdTest(input_size_, filter_size_, stride_, padding_)))
        setattr(Conv2DTest, 'testInceptionFwdDilatedConv_' + str(index), test_util.run_in_graph_and_eager_modes(GetInceptionFwdDilatedConvTest(input_size_, filter_size_, stride_, padding_)))
        setattr(Conv2DTest, 'testInceptionBackInput_' + str(index), test_util.run_in_graph_and_eager_modes(GetInceptionBackInputTest(input_size_, filter_size_, output_size_, stride_, padding_)))
        setattr(Conv2DTest, 'testInceptionBackFilter_' + str(index), test_util.run_in_graph_and_eager_modes(GetInceptionBackFilterTest(input_size_, filter_size_, output_size_, [stride_, stride_], padding_)))
    ishape = [1, 400, 400, 1]
    fshape = [1, 1, 1, 256]
    oshape = [1, 400, 400, 256]
    setattr(Conv2DTest, 'testInceptionFwd_No_Winograd_Nonfused', test_util.run_in_graph_and_eager_modes(GetInceptionFwdTest(ishape, fshape, 1, 'SAME', gpu_only=True)))
    setattr(Conv2DTest, 'testInceptionFwdDilatedConv_No_Winograd_Nonfused', test_util.run_in_graph_and_eager_modes(GetInceptionFwdDilatedConvTest(ishape, fshape, 1, 'SAME')))
    setattr(Conv2DTest, 'testInceptionBackInput_No_Winograd_Nonfused', test_util.run_in_graph_and_eager_modes(GetInceptionBackInputTest(ishape, fshape, oshape, 1, 'SAME', gpu_only=True)))
    setattr(Conv2DTest, 'testInceptionBackFilter_No_Winograd_Nonfused', test_util.run_in_graph_and_eager_modes(GetInceptionBackFilterTest(ishape, fshape, oshape, [1, 1], 'SAME', gpu_only=True)))
    test.main()