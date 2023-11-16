import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn

class TRTReduceSumTest(InferencePassTest):

    def setUp(self):
        if False:
            return 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 10, 192], dtype='float32')
            reduce_sum = paddle.sum(data, axis=[2, -1], keepdim=True)
            out = nn.batch_norm(reduce_sum, is_test=True)
        self.feeds = {'data': np.random.random([3, 3, 10, 192]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTReduceSumTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTReduceSumTest.DynamicShapeParam({'data': [1, 3, 8, 8]}, {'data': [3, 3, 10, 192]}, {'data': [3, 3, 10, 192]}, False)

    def test_check_output(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TRTReduceSumAllTest(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 10, 192], dtype='float32')
            reduce_sum = paddle.sum(data, keepdim=True)
            out = nn.batch_norm(reduce_sum, is_test=True)
        self.feeds = {'data': np.random.random([3, 3, 10, 192]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTReduceSumAllTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTReduceSumAllTest.DynamicShapeParam({'data': [1, 3, 8, 8]}, {'data': [3, 3, 10, 192]}, {'data': [3, 3, 10, 192]}, False)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
if __name__ == '__main__':
    unittest.main()