import os
import shutil
import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker

class SkipLayernormFusePassTest0(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(name='data1', shape=[-1, 3, 128, 128], dtype='float32')
            data2 = paddle.static.data(name='data2', shape=[-1, 3, 128, 128], dtype='float32')
            eltwise_out = self.append_eltwise(data1, data2)
            out = paddle.nn.functional.layer_norm(eltwise_out, eltwise_out.shape[1:])
        self.feeds = {'data1': np.random.random([1, 3, 128, 128]).astype('float32'), 'data2': np.random.random([1, 3, 128, 128]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = SkipLayernormFusePassTest0.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False)
        self.dynamic_shape_params = SkipLayernormFusePassTest0.DynamicShapeParam({'data1': [1, 1, 1, 128], 'data2': [1, 1, 1, 128]}, {'data1': [1, 3, 128, 128], 'data2': [1, 3, 128, 128]}, {'data1': [1, 3, 128, 128], 'data2': [1, 3, 128, 128]}, False)
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        if False:
            for i in range(10):
                print('nop')
        return paddle.add(data1, data2)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=0.01, rtol=1e-05)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class SkipLayernormFusePassTest1(InferencePassTest):

    def setUp(self):
        if False:
            return 10
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(name='data1', shape=[-1, 256, 1536], dtype='float32')
            data2 = paddle.static.data(name='data2', shape=[-1, 256, 1536], dtype='float32')
            eltwise_out = self.append_eltwise(data1, data2)
            out = paddle.nn.functional.layer_norm(eltwise_out, eltwise_out.shape[1:])
        self.feeds = {'data1': np.random.random([1, 256, 1536]).astype('float32'), 'data2': np.random.random([1, 256, 1536]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = SkipLayernormFusePassTest1.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False)
        self.dynamic_shape_params = SkipLayernormFusePassTest1.DynamicShapeParam({'data1': [1, 1, 1], 'data2': [1, 1, 1]}, {'data1': [1, 384, 1536], 'data2': [1, 384, 1536]}, {'data1': [1, 384, 1536], 'data2': [1, 384, 1536]}, False)
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        if False:
            for i in range(10):
                print('nop')
        return paddle.add(data1, data2)

    def test_check_output(self):
        if False:
            return 10
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=0.01, rtol=1e-05)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class SkipLayernormFusePassTest2(InferencePassTest):

    def setUp(self):
        if False:
            return 10
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(name='data1', shape=[-1, 128, 64, 768], dtype='float32')
            data2 = paddle.static.data(name='data2', shape=[-1, 128, 64, 768], dtype='float32')
            eltwise_out = self.append_eltwise(data1, data2)
            out = paddle.nn.functional.layer_norm(eltwise_out, eltwise_out.shape[1:])
        self.feeds = {'data1': np.random.random([1, 128, 64, 768]).astype('float32'), 'data2': np.random.random([1, 128, 64, 768]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = SkipLayernormFusePassTest2.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False)
        self.dynamic_shape_params = SkipLayernormFusePassTest2.DynamicShapeParam({'data1': [1, 1, 1, 1], 'data2': [1, 1, 1, 1]}, {'data1': [1, 128, 64, 768], 'data2': [1, 128, 64, 768]}, {'data1': [1, 128, 64, 768], 'data2': [1, 128, 64, 768]}, False)
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        if False:
            while True:
                i = 10
        return paddle.add(data1, data2)

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=0.1, rtol=1e-05)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class SkipLayernormFusePassTest3(InferencePassTest):

    def setUp(self):
        if False:
            return 10
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(name='data1', shape=[-1, 128, 128], dtype='float32')
            data2 = paddle.static.data(name='data2', shape=[-1, 128, 128], dtype='float32')
            eltwise_out = self.append_eltwise(data1, data2)
            out = paddle.nn.functional.layer_norm(eltwise_out, eltwise_out.shape[1:])
        self.feeds = {'data1': np.random.random([1, 128, 128]).astype('float32'), 'data2': np.random.random([1, 128, 128]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = SkipLayernormFusePassTest3.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False)
        self.dynamic_shape_params = SkipLayernormFusePassTest3.DynamicShapeParam({'data1': [1, 1, 1], 'data2': [1, 1, 1]}, {'data1': [1, 128, 128], 'data2': [1, 128, 128]}, {'data1': [1, 128, 128], 'data2': [1, 128, 128]}, False)
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        if False:
            for i in range(10):
                print('nop')
        return paddle.add(data1, data2)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=0.1, rtol=1e-05)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
if __name__ == '__main__':
    unittest.main()