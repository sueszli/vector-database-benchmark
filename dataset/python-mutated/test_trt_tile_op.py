import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker

class TRTTileTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[4, 3, 224, 256], dtype='float32')
            tile_out = paddle.tile(x=data, repeat_times=[1, 1, 1, 1])
            out = paddle.static.nn.batch_norm(tile_out, is_test=True)
        self.feeds = {'data': np.random.random([4, 3, 224, 256]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTTileTest.TensorRTParam(1 << 30, 16, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TRTTileExpandTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[1, 1, 1, 1], dtype='float32')
            tile_out = paddle.tile(x=data, repeat_times=[1, 4, 1080, 1920])
            out = paddle.static.nn.batch_norm(tile_out, is_test=True)
        self.feeds = {'data': np.random.random([1, 1, 1, 1]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTTileExpandTest.TensorRTParam(1 << 30, 1, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TRTTileExpandStaticTest(InferencePassTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[1, 1, 1, 1], dtype='float32')
            tile_out = paddle.tile(x=data, repeat_times=[1, 4, 1080, 1920])
            out = paddle.static.nn.batch_norm(tile_out, is_test=True)
        self.feeds = {'data': np.random.random([1, 1, 1, 1]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTTileExpandStaticTest.TensorRTParam(1 << 30, 1, 1, AnalysisConfig.Precision.Float32, True, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TRTTileExpandHalfTest(InferencePassTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[1, 1, 1, 1], dtype='float32')
            tile_out = paddle.tile(x=data, repeat_times=[1, 4, 1080, 1920])
            out = paddle.static.nn.batch_norm(tile_out, is_test=True)
        self.feeds = {'data': np.random.random([1, 1, 1, 1]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTTileExpandHalfTest.TensorRTParam(1 << 30, 1, 1, AnalysisConfig.Precision.Half, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, 0.0001, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
if __name__ == '__main__':
    unittest.main()