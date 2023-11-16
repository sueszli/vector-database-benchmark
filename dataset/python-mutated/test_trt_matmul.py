import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn

class TensorRTMatMulDims2Test(InferencePassTest):

    def setUp(self):
        if False:
            return 10
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[24, 24], dtype='float32')
            matmul_out = paddle.matmul(x=data, y=data, transpose_x=self.transpose_x, transpose_y=self.transpose_y)
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)
        self.feeds = {'data': np.ones([24, 24]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulDims2Test.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        if False:
            print('Hello World!')
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 2.0

    def test_check_output(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTMatMulTest(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, 24, 24], dtype='float32')
            matmul_out = paddle.matmul(x=data, y=data, transpose_x=self.transpose_x, transpose_y=self.transpose_y)
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)
        self.feeds = {'data': np.ones([1, 6, 24, 24]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTMatMulTransposeXTest(TensorRTMatMulTest):

    def set_params(self):
        if False:
            for i in range(10):
                print('nop')
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 1.0

class TensorRTMatMulTransposeYTest(TensorRTMatMulTest):

    def set_params(self):
        if False:
            print('Hello World!')
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 1.0

class TensorRTMatMulScaleTest(TensorRTMatMulTest):

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 2.0

class TensorRTMatMulBroadcastTest(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_params()
        place = base.CPUPlace()
        with base.program_guard(self.main_program, self.startup_program):
            data_x = paddle.static.data(name='data_x', shape=[-1, 6, 24], dtype='float32')
            data_y = paddle.static.data(name='data_y', shape=[24, 16], dtype='float32')
            matmul_out = paddle.matmul(x=data_x, y=data_y, transpose_x=self.transpose_x, transpose_y=self.transpose_y)
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)
        self.feeds = {'data_x': np.ones([2, 6, 24]).astype('float32'), 'data_y': np.ones([24, 16]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulBroadcastTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        if False:
            while True:
                i = 10
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
if __name__ == '__main__':
    unittest.main()