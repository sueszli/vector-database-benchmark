import os
import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

class TensorRTSubgraphPassConvTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, 64, 64], dtype='float32')
            conv_out = paddle.static.nn.conv2d(input=data, num_filters=self.conv_num_filters, filter_size=self.conv_filter_size, groups=self.conv_groups, padding=self.conv_padding, bias_attr=False, use_cudnn=self.use_cudnn, act=None)
        self.feeds = {'data': np.random.random([1, 6, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConvTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [conv_out]

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = [1, 1]
        self.use_cudnn = True

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassConvValidPaddingTest(TensorRTSubgraphPassConvTest):

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = 'VALID'
        self.use_cudnn = True

class TensorRTSubgraphPassConvSamePaddingTest(InferencePassTest):

    def set_params(self):
        if False:
            print('Hello World!')
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = 'SAME'
        self.use_cudnn = True

class TensorRTSubgraphPassDepthwiseConvTest(TensorRTSubgraphPassConvTest):

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 6
        self.conv_padding = [1, 1]
        self.use_cudnn = False

class TensorRTSubgraphPassDepthwiseConv2Test(TensorRTSubgraphPassConvTest):

    def set_params(self):
        if False:
            for i in range(10):
                print('nop')
        self.conv_num_filters = 12
        self.conv_filter_size = 6
        self.conv_groups = 6
        self.conv_padding = [1, 1]
        self.use_cudnn = False

class TensorRTSubgraphPassConvTransposeTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, 64, 64], dtype='float32')
            conv_out = paddle.static.nn.conv2d_transpose(input=data, num_filters=self.conv_num_filters, filter_size=self.conv_filter_size, groups=self.conv_groups, padding=self.conv_padding, bias_attr=False, use_cudnn=self.use_cudnn, act=None)
        self.feeds = {'data': np.random.random([1, 6, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConvTransposeTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [conv_out]

    def set_params(self):
        if False:
            return 10
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = [1, 1]
        self.use_cudnn = True

    def test_check_output(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassConvTransposeValidPaddingTest(TensorRTSubgraphPassConvTransposeTest):

    def set_params(self):
        if False:
            print('Hello World!')
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = 'VALID'
        self.use_cudnn = True

class TensorRTSubgraphPassConvTransposeSamePaddingTest(TensorRTSubgraphPassConvTransposeTest):

    def set_params(self):
        if False:
            return 10
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = 'SAME'
        self.use_cudnn = True

class TensorRTSubgraphPassConvTransposeMultiGroupTest(TensorRTSubgraphPassConvTransposeTest):

    def set_params(self):
        if False:
            print('Hello World!')
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 2
        self.conv_padding = [1, 1]
        self.use_cudnn = True

class TensorRTSubgraphPassConvTranspose2Test(TensorRTSubgraphPassConvTransposeTest):

    def set_params(self):
        if False:
            return 10
        self.conv_num_filters = 12
        self.conv_filter_size = 4
        self.conv_groups = 6
        self.conv_padding = [1, 1]
        self.use_cudnn = False

class TensorRTSubgraphPassDepthwiseConvTransposeTest(TensorRTSubgraphPassConvTransposeTest):

    def set_params(self):
        if False:
            for i in range(10):
                print('nop')
        self.conv_num_filters = 6
        self.conv_filter_size = 4
        self.conv_groups = 6
        self.conv_padding = [1, 1]
        self.use_cudnn = False

class DynamicShapeTensorRTSubgraphPassConvTest(InferencePassTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, -1, -1], dtype='float32')
            conv_out = paddle.static.nn.conv2d(input=data, num_filters=self.conv_num_filters, filter_size=self.conv_filter_size, groups=self.conv_groups, padding=self.conv_padding, bias_attr=False, use_cudnn=self.use_cudnn, stride=self.stride, act=None)
        self.feeds = {'data': np.random.random([32, 6, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = DynamicShapeTensorRTSubgraphPassConvTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = DynamicShapeTensorRTSubgraphPassConvTest.DynamicShapeParam({'conv2d_0.tmp_0': [1, 6, 8, 8], 'data': [1, 6, 8, 8], 'depthwise_conv2d_0.tmp_0': [1, 6, 8, 8]}, {'conv2d_0.tmp_0': [32, 6, 64, 64], 'data': [32, 6, 64, 64], 'depthwise_conv2d_0.tmp_0': [32, 6, 64, 64]}, {'conv2d_0.tmp_0': [16, 6, 16, 16], 'data': [16, 6, 16, 16], 'depthwise_conv2d_0.tmp_0': [16, 6, 16, 16]}, False)
        self.fetch_list = [conv_out]

    def set_params(self):
        if False:
            return 10
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 6
        self.conv_padding = 'SAME'
        self.use_cudnn = True
        self.stride = [2, 2]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class DynamicShapeTensorRTSubgraphPassDepthwiseConvTransposeTest(DynamicShapeTensorRTSubgraphPassConvTest):

    def set_params(self):
        if False:
            print('Hello World!')
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 6
        self.conv_padding = 'SAME'
        self.use_cudnn = False
        self.stride = [2, 2]
if __name__ == '__main__':
    unittest.main()