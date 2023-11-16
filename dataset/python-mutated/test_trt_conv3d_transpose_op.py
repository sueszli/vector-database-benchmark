import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker

class TensorRTSubgraphPassConv3dTransposeTest(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 4, 4, 32, 32], dtype='float32')
            conv_out = paddle.static.nn.conv3d_transpose(input=data, num_filters=self.conv_num_filters, filter_size=self.conv_filter_size, groups=self.conv_groups, padding=self.conv_padding, bias_attr=False, use_cudnn=self.use_cudnn, stride=1, act=None)
        self.feeds = {'data': np.random.random([1, 4, 4, 32, 32]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConv3dTransposeTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [conv_out]

    def set_params(self):
        if False:
            return 10
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = [1, 1, 1]
        self.use_cudnn = True

    def test_check_output(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassConv3dTransposeSamePaddingTest(TensorRTSubgraphPassConv3dTransposeTest):

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = 'VALID'
        self.use_cudnn = True

class TensorRTSubgraphPassConv3dTransposeMultigroupTest(TensorRTSubgraphPassConv3dTransposeTest):

    def set_params(self):
        if False:
            print('Hello World!')
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 2
        self.conv_padding = 'VALID'
        self.use_cudnn = True

class DynamicShapeTensorRTSubgraphPassConv3dTransposeTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, -1, -1, -1], dtype='float32')
            conv_out = paddle.static.nn.conv3d_transpose(input=data, num_filters=self.conv_num_filters, filter_size=self.conv_filter_size, groups=self.conv_groups, padding=self.conv_padding, bias_attr=False, use_cudnn=self.use_cudnn, stride=self.stride, act=None)
        self.feeds = {'data': np.random.random([1, 6, 32, 32, 8]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = DynamicShapeTensorRTSubgraphPassConv3dTransposeTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = DynamicShapeTensorRTSubgraphPassConv3dTransposeTest.DynamicShapeParam({'data': [1, 6, 8, 8, 8], 'conv3d_transpose_0.tmp_0': [1, 6, 8, 8, 1]}, {'data': [32, 6, 32, 32, 8], 'conv3d_transpose_0.tmp_0': [32, 6, 64, 64, 16]}, {'data': [16, 6, 16, 16, 8], 'conv3d_transpose_0.tmp_0': [16, 6, 16, 16, 8]}, False)
        self.fetch_list = [conv_out]

    def set_params(self):
        if False:
            for i in range(10):
                print('nop')
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 6
        self.conv_padding = 'SAME'
        self.use_cudnn = True
        self.stride = [2, 2, 2]

    def test_check_output(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
if __name__ == '__main__':
    unittest.main()