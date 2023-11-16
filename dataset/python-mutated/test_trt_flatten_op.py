import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn

class TRTFlattenTest(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, 64, 64], dtype='float32')
            flatten_out = self.append_flatten(data)
            out = nn.batch_norm(flatten_out, is_test=True)
        self.feeds = {'data': np.random.random([1, 6, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTFlattenTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_flatten(self, data):
        if False:
            while True:
                i = 10
        return paddle.flatten(data, 1, -1)

    def test_check_output(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TRTFlattenDynamicTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, 64, 64], dtype='float32')
            flatten_out = self.append_flatten(data)
            out = nn.batch_norm(flatten_out, is_test=True)
        self.feeds = {'data': np.random.random([2, 6, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTFlattenDynamicTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = TRTFlattenDynamicTest.DynamicShapeParam({'data': [2, 6, 64, 64], 'flatten_0.tmp_0': [2, 6 * 64 * 64]}, {'data': [2, 6, 64, 64], 'flatten_0.tmp_0': [2, 6 * 64 * 64]}, {'data': [2, 6, 64, 64], 'flatten_0.tmp_0': [2, 6 * 64 * 64]}, False)
        self.fetch_list = [out]

    def append_flatten(self, data):
        if False:
            while True:
                i = 10
        return paddle.flatten(data, 1, -1)

    def test_check_output(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
if __name__ == '__main__':
    unittest.main()