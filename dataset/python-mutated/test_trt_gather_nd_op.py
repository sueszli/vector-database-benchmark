import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn

class TRTGatherNdTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 4], dtype='float32')
            index = paddle.static.data(name='index', shape=[-1, 2, 2], dtype='int32')
            gather_nd = paddle.gather_nd(data, index)
            out = nn.batch_norm(gather_nd, is_test=True)
        self.feeds = {'data': np.random.random([2, 3, 4]).astype('float32'), 'index': np.array([[[0, 1], [1, 0]], [[1, 2], [0, 1]]]).astype('int32')}
        self.enable_trt = True
        self.trt_parameters = TRTGatherNdTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTGatherNdTest.DynamicShapeParam({'data': [1, 3, 4], 'index': [1, 2, 2]}, {'data': [3, 3, 4], 'index': [3, 2, 2]}, {'data': [3, 3, 4], 'index': [3, 2, 2]}, False)

    def test_check_output(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TRTGatherNdFp16Test(InferencePassTest):

    def setUp(self):
        if False:
            return 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 1280, 192], dtype='float32')
            index = paddle.static.data(name='index', shape=[-1, 1028, 2], dtype='int32')
            gather_nd = paddle.gather_nd(data, index)
            out = nn.batch_norm(gather_nd, is_test=True)
        index_data = np.zeros((1, 1028, 2), dtype='int32')
        self.feeds = {'data': np.random.random([1, 1280, 192]).astype('float32'), 'index': index_data}
        self.enable_trt = True
        self.trt_parameters = TRTGatherNdFp16Test.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Half, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTGatherNdFp16Test.DynamicShapeParam({'data': [1, 1280, 192], 'index': [1, 1028, 2]}, {'data': [3, 1280, 192], 'index': [3, 1028, 2]}, {'data': [3, 1280, 192], 'index': [3, 1028, 2]}, False)

    def test_check_output(self, atol=0.001):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
if __name__ == '__main__':
    unittest.main()