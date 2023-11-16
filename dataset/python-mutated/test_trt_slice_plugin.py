import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig
from paddle.static import nn

class SlicePluginTRTTest(InferencePassTest):

    def setUpSliceParams(self):
        if False:
            print('Hello World!')
        self.params_axes = [1, 3]
        self.params_starts = [0, 1]
        self.params_ends = [2, 3]

    def setUpTensorRTParams(self):
        if False:
            while True:
                i = 10
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.enable_trt = True

    def setUp(self):
        if False:
            print('Hello World!')
        self.setUpSliceParams()
        self.setUpTensorRTParams()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[3, 3, 3, 3], dtype='float32')
            axes = self.params_axes
            starts = self.params_starts
            ends = self.params_ends
            slice_out = paddle.slice(data, axes=axes, starts=starts, ends=ends)
            out = nn.batch_norm(slice_out, is_test=True)
        self.feeds = {'data': np.random.random((3, 3, 3, 3)).astype('float32')}
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            atol = 1e-05
            if self.trt_parameters.precision == AnalysisConfig.Precision.Half:
                atol = 0.001
            self.check_output_with_option(use_gpu[i], atol)

class SlicePluginTRTTestNegativeStartsAndEnds(SlicePluginTRTTest):

    def setUpSliceParams(self):
        if False:
            i = 10
            return i + 15
        self.params_axes = [2, 3]
        self.params_starts = [-3, -2]
        self.params_ends = [-1, 3]

class SlicePluginTRTTestStartsAndEndsBoundCheck(SlicePluginTRTTest):

    def setUpSliceParams(self):
        if False:
            for i in range(10):
                print('nop')
        self.params_axes = [2, 3]
        self.params_starts = [-5, -2]
        self.params_ends = [-1, 8]

class SlicePluginTRTTestFp16(SlicePluginTRTTest):

    def setUpTensorRTParams(self):
        if False:
            print('Hello World!')
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Half, False, False)
        self.enable_trt = True

class StaticSlicePluginTRTTestFp16(SlicePluginTRTTest):

    def setUpTensorRTParams(self):
        if False:
            i = 10
            return i + 15
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Half, True, False)
        self.enable_trt = True

class StaticSlicePluginTRTTestFp32(SlicePluginTRTTest):

    def setUpTensorRTParams(self):
        if False:
            while True:
                i = 10
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, True, False)
        self.enable_trt = True

class SlicePluginTRTTestInt32(SlicePluginTRTTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setUpSliceParams()
        self.setUpTensorRTParams()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[3, 3, 3, 3], dtype='int32')
            axes = self.params_axes
            starts = self.params_starts
            ends = self.params_ends
            slice_out = paddle.slice(data, axes=axes, starts=starts, ends=ends)
            cast_out = paddle.cast(slice_out, 'float32')
            out = nn.batch_norm(cast_out, is_test=True)
        self.feeds = {'data': np.random.random((3, 3, 3, 3)).astype('int32')}
        self.fetch_list = [out]

class StaticSlicePluginTRTTestInt32(SlicePluginTRTTest):

    def setUpTensorRTParams(self):
        if False:
            while True:
                i = 10
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, True, False)
        self.enable_trt = True

    def setUp(self):
        if False:
            return 10
        self.setUpSliceParams()
        self.setUpTensorRTParams()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[3, 3, 3, 3], dtype='int32')
            axes = self.params_axes
            starts = self.params_starts
            ends = self.params_ends
            slice_out = paddle.slice(data, axes=axes, starts=starts, ends=ends)
            cast_out = paddle.cast(slice_out, 'float32')
            out = nn.batch_norm(cast_out, is_test=True)
        self.feeds = {'data': np.random.random((3, 3, 3, 3)).astype('int32')}
        self.fetch_list = [out]
if __name__ == '__main__':
    unittest.main()