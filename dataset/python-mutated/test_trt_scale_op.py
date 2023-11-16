import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn

class TRTScaleTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 512], dtype='float32')
            scale_out = self.append_scale(data)
            out = nn.batch_norm(scale_out, is_test=True)
        self.feeds = {'data': np.random.random([1, 512]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTScaleTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_scale(self, data):
        if False:
            return 10
        return paddle.scale(x=data, scale=2.0, bias=-1.0, bias_after_scale=False)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TRTScaleShape2Test(InferencePassTest):

    def setUp(self):
        if False:
            return 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 512, 512], dtype='float32')
            scale_out = self.append_scale(data)
            out = nn.batch_norm(scale_out, is_test=True)
        self.feeds = {'data': np.random.random([1, 512, 512]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTScaleShape2Test.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_scale(self, data):
        if False:
            i = 10
            return i + 15
        return paddle.scale(x=data, scale=2.0, bias=-1.0, bias_after_scale=False)

    def test_check_output(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
if __name__ == '__main__':
    unittest.main()