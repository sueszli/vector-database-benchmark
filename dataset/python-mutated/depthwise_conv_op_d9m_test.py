"""Functional tests for determinsitic depthwise convolutional operations."""
from tensorflow.python.eager import backprop
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.nn_ops import depthwise_conv_op_base
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.nn_grad import _DepthwiseConv2dNativeBackpropFilterGrad
from tensorflow.python.ops.nn_grad import _DepthwiseConv2dNativeBackpropInputGrad
from tensorflow.python.platform import test

@test_util.run_all_without_tensor_float_32('Uses matmul')
class DepthwiseConv2DDeterministicTest(depthwise_conv_op_base.DepthwiseConv2DBase):
    """Test determinism-related functionality of tf.nn.depthwise_conv2d."""

    def _genParams(self, use_cudnn=False, data_format='NHWC', dtype=dtypes.float32, seed=123):
        if False:
            i = 10
            return i + 15
        random_seed.set_seed(seed)
        batch_size = 2
        if use_cudnn:
            input_channels = 1
        else:
            input_channels = 2
        input_height = 500
        input_width = 1000
        if data_format == 'NHWC':
            input_shape = (batch_size, input_height, input_width, input_channels)
        else:
            input_shape = (batch_size, input_channels, input_height, input_width)
        input_data = random_ops.random_normal(input_shape, dtype=dtype)
        filter_height = 7
        filter_width = 7
        channel_multiplier = 10
        filter_shape = (filter_height, filter_width, input_channels, channel_multiplier)
        filter_data = random_ops.random_normal(filter_shape, dtype=dtype)
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        output_height = input_height
        output_width = input_width
        output_channels = input_channels * channel_multiplier
        if data_format == 'NHWC':
            output_shape = (batch_size, output_height, output_width, output_channels)
        else:
            output_shape = (batch_size, output_channels, output_height, output_width)
        return (input_data, filter_data, strides, padding, output_shape)

    def _testForwardDeterminismCase(self, use_cudnn=False, data_format='NHWC', dtype=dtypes.float32):
        if False:
            for i in range(10):
                print('nop')
        for seed in range(5):
            p = self._genParams(use_cudnn, data_format, dtype, seed=seed)
            (input_data, filter_data, strides, padding, _) = p
            result_a = nn_impl.depthwise_conv2d_v2(input_data, filter_data, strides, padding, data_format)
            result_b = nn_impl.depthwise_conv2d_v2(input_data, filter_data, strides, padding, data_format)
            self.assertAllEqual(result_a, result_b)

    @test_util.run_gpu_only
    def testForwardDeterminismGPU(self):
        if False:
            print('Hello World!')
        for use_cudnn in [False, True]:
            for data_format in ['NHWC', 'NCHW']:
                for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
                    self._testForwardDeterminismCase(use_cudnn, data_format, dtype=dtype)

    def testForwardDeterminismCPU(self):
        if False:
            return 10
        if tf_config.list_physical_devices('GPU'):
            self.skipTest('Test only runs when there is no GPU')
        data_format = 'NHWC'
        for dtype in [dtypes.bfloat16.as_numpy_dtype, dtypes.float32, dtypes.float64]:
            self._testForwardDeterminismCase(data_format=data_format, dtype=dtype)

    def _testBackwardDeterminismCase(self, using_gpu=False, use_cudnn=False, data_format='NHWC', dtype=dtypes.float32):
        if False:
            i = 10
            return i + 15
        p = self._genParams(use_cudnn, data_format, dtype, seed=123)
        (input_data, filter_data, strides, padding, output_shape) = p

        def Gradients(upstream_gradients):
            if False:
                while True:
                    i = 10
            with backprop.GradientTape() as tape:
                tape.watch(input_data)
                tape.watch(filter_data)
                op_output = nn_impl.depthwise_conv2d_v2(input_data, filter_data, strides, padding, data_format)
                gradient_injector_output = op_output * upstream_gradients
            return tape.gradient(gradient_injector_output, [input_data, filter_data])
        for seed in (987, 988):
            upstream_gradients = random_ops.random_normal(output_shape, dtype=dtype, seed=seed)
            (input_gradients_a, filter_gradients_a) = Gradients(upstream_gradients)
            (input_gradients_b, filter_gradients_b) = Gradients(upstream_gradients)
            self.assertAllEqual(input_gradients_a, input_gradients_b)
            self.assertAllEqual(filter_gradients_a, filter_gradients_b)

    @test_util.run_gpu_only
    def testBackwardDeterminismGPU(self):
        if False:
            return 10
        using_gpu = True
        for use_cudnn in [False, True]:
            for data_format in ['NHWC', 'NCHW']:
                for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
                    self._testBackwardDeterminismCase(using_gpu, use_cudnn, data_format, dtype)

    def testBackwardDeterminismCPU(self):
        if False:
            while True:
                i = 10
        if tf_config.list_physical_devices('GPU'):
            self.skipTest('Test only runs when there is no GPU')
        data_format = 'NHWC'
        for dtype in [dtypes.bfloat16.as_numpy_dtype, dtypes.float32, dtypes.float64]:
            self._testBackwardDeterminismCase(data_format=data_format, dtype=dtype)
if __name__ == '__main__':
    tf_config.enable_op_determinism()
    test.main()