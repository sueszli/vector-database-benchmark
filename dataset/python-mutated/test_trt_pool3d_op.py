import itertools
import os
import shutil
import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker

class TensorRTPool3dTest(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.bs = 1
        self.channel = 3
        self.depth = 8
        self.height = 8
        self.width = 8
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.ceil_mode = False
        self.exclusive = False
        self.enable_trt = True
        self.serialize = False
        self.precision = AnalysisConfig.Precision.Float32
        self.feeds = {'data': np.random.random([self.bs, self.channel, self.depth, self.height, self.width]).astype('float32')}

    def set_extra_config(self):
        if False:
            while True:
                i = 10
        pass

    def build_network(self):
        if False:
            return 10
        self.set_extra_config()
        self.trt_parameters = TensorRTPool3dTest.TensorRTParam(1 << 30, self.bs, 0, self.precision, self.serialize, False)
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, self.channel, self.depth, self.height, self.width], dtype='float32')
            if self.pool_type == 'max':
                pool_out = paddle.nn.functional.max_pool3d(x=data, kernel_size=self.pool_size, stride=self.pool_stride, padding=self.pool_padding, ceil_mode=self.ceil_mode)
            else:
                pool_out = paddle.nn.functional.avg_pool3d(x=data, kernel_size=self.pool_size, stride=self.pool_stride, padding=self.pool_padding, ceil_mode=self.ceil_mode, exclusive=self.exclusive)
            self.fetch_list = [pool_out]

    def check_output(self):
        if False:
            print('Hello World!')
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            if self.precision == AnalysisConfig.Precision.Float32:
                (atol, rtol) = (1e-05, 1e-05)
            elif self.precision == AnalysisConfig.Precision.Half:
                (atol, rtol) = (0.001, 0.001)
            else:
                raise ValueError(f'Unsupported precision {self.precision}')
            self.check_output_with_option(use_gpu, atol=atol, rtol=rtol)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def run_test(self):
        if False:
            while True:
                i = 10
        self.build_network()
        self.check_output()

    def test(self):
        if False:
            return 10
        precision_options = [AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half]
        serialize_options = [False, True]
        dynamic_shape_profile = InferencePassTest.DynamicShapeParam({'data': [self.bs, self.channel, self.depth // 2, self.height // 2, self.width // 2]}, {'data': [self.bs, self.channel, self.depth, self.height, self.width]}, {'data': [self.bs, self.channel, self.depth, self.height, self.width]}, False)
        dynamic_shape_options = [None, dynamic_shape_profile]
        for (precision, serialize, dynamic_shape) in itertools.product(precision_options, serialize_options, dynamic_shape_options):
            is_dynamic = True if dynamic_shape_options is not None else False
            with self.subTest(f'Precision: {precision}, Serialize: {serialize}, Dynamic: {is_dynamic}'):
                self.precision = precision
                self.serialize = serialize
                self.dynamic_shape_params = dynamic_shape
                self.run_test()

class TensorRTAvgPool3dTest(TensorRTPool3dTest):

    def set_extra_config(self):
        if False:
            i = 10
            return i + 15
        self.pool_size = 2
        self.pool_type = 'avg'
        self.pool_stride = 1
        self.pool_padding = 0
        self.ceil_mode = False
        self.exclusive = False

class TensorRTAdaptiveAvgPool3DTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.bs = 1
        self.channel = 3
        self.depth = 8
        self.height = 8
        self.width = 8
        self.enable_trt = True
        self.serialize = False
        self.precision = AnalysisConfig.Precision.Float32
        self.feeds = {'data': np.random.random([self.bs, self.channel, self.depth, self.height, self.width]).astype('float32')}

    def build_network(self):
        if False:
            print('Hello World!')
        self.trt_parameters = TensorRTPool3dTest.TensorRTParam(1 << 30, self.bs, 0, self.precision, self.serialize, False)
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, self.channel, self.depth, self.height, self.width], dtype='float32')
            pool_out = paddle.nn.functional.adaptive_avg_pool3d(x=data, output_size=[3, 3, 3])
            self.fetch_list = [pool_out]

    def check_output(self):
        if False:
            i = 10
            return i + 15
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def run_test(self):
        if False:
            i = 10
            return i + 15
        self.build_network()
        self.check_output()

    def test(self):
        if False:
            while True:
                i = 10
        precision_options = [AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half]
        serialize_options = [False, True]
        dynamic_shape_profile = InferencePassTest.DynamicShapeParam({'data': [self.bs, self.channel, self.depth // 2, self.height // 2, self.width // 2]}, {'data': [self.bs, self.channel, self.depth, self.height, self.width]}, {'data': [self.bs, self.channel, self.depth, self.height, self.width]}, False)
        dynamic_shape_options = [None, dynamic_shape_profile]
        for (precision, serialize, dynamic_shape) in itertools.product(precision_options, serialize_options, dynamic_shape_options):
            is_dynamic = True if dynamic_shape_options is not None else False
            with self.subTest(f'Precision: {precision}, Serialize: {serialize}, Dynamic: {is_dynamic}'):
                self.precision = precision
                self.serialize = serialize
                self.dynamic_shape_params = dynamic_shape
                self.run_test()

class TensorRTAdaptiveMaxPool3DTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.bs = 1
        self.channel = 3
        self.depth = 8
        self.height = 8
        self.width = 8
        self.enable_trt = True
        self.serialize = False
        self.precision = AnalysisConfig.Precision.Float32
        self.feeds = {'data': np.random.random([self.bs, self.channel, self.depth, self.height, self.width]).astype('float32')}

    def build_network(self):
        if False:
            print('Hello World!')
        self.trt_parameters = TensorRTPool3dTest.TensorRTParam(1 << 30, self.bs, 0, self.precision, self.serialize, False)
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, self.channel, self.depth, self.height, self.width], dtype='float32')
            pool_out = paddle.nn.functional.adaptive_max_pool3d(x=data, output_size=[3, 3, 3])
            self.fetch_list = [pool_out]

    def check_output(self):
        if False:
            while True:
                i = 10
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def run_test(self):
        if False:
            while True:
                i = 10
        self.build_network()
        self.check_output()

    def test(self):
        if False:
            print('Hello World!')
        precision_options = [AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half]
        serialize_options = [False, True]
        dynamic_shape_profile = InferencePassTest.DynamicShapeParam({'data': [self.bs, self.channel, self.depth // 2, self.height // 2, self.width // 2]}, {'data': [self.bs, self.channel, self.depth, self.height, self.width]}, {'data': [self.bs, self.channel, self.depth, self.height, self.width]}, False)
        dynamic_shape_options = [None, dynamic_shape_profile]
        for (precision, serialize, dynamic_shape) in itertools.product(precision_options, serialize_options, dynamic_shape_options):
            is_dynamic = True if dynamic_shape_options is not None else False
            with self.subTest(f'Precision: {precision}, Serialize: {serialize}, Dynamic: {is_dynamic}'):
                self.precision = precision
                self.serialize = serialize
                self.dynamic_shape_params = dynamic_shape
                self.run_test()
if __name__ == '__main__':
    unittest.main()