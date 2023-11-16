import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig

class FCFusePassTRTTest(InferencePassTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[32, 128, 2, 2], dtype='float32')
            fc_out1 = paddle.static.nn.fc(x=data, size=128, num_flatten_dims=1, activation='relu')
            out = paddle.nn.functional.softmax(fc_out1)
        self.feeds = {'data': np.random.random((32, 128, 2, 2)).astype('float32')}
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            while True:
                i = 10
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol=0.0001, rtol=0.001)

class FCFusePassTRTStaticDims4Cols1Test(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[32, 128, 32, 8], dtype='float32')
            fc_out1 = paddle.static.nn.fc(x=data, size=64, num_flatten_dims=1, activation='relu')
            out = paddle.nn.functional.softmax(fc_out1)
        self.feeds = {'data': np.random.random((32, 128, 32, 8)).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTStaticDims4Cols1Test.TensorRTParam(1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            return 10
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol=0.0001, rtol=0.001)

class FCFusePassTRTStaticDims4Cols2Test(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[3, 24, 16, 16], dtype='float32')
            fc_out1 = paddle.static.nn.fc(x=data, size=32, num_flatten_dims=2, activation='relu')
            out = paddle.nn.functional.softmax(fc_out1)
        self.feeds = {'data': np.random.random((3, 24, 16, 16)).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTStaticDims4Cols2Test.TensorRTParam(1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol=0.0001, rtol=0.001)

class FCFusePassTRTDynamicDims2Test(InferencePassTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[32, 128], dtype='float32')
            fc_out1 = paddle.static.nn.fc(x=data, size=64, num_flatten_dims=1, activation='relu')
            out = paddle.nn.functional.softmax(fc_out1)
        self.feeds = {'data': np.random.random((32, 128)).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims2Test.TensorRTParam(1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims2Test.DynamicShapeParam({'data': [1, 128]}, {'data': [64, 128]}, {'data': [32, 128]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            print('Hello World!')
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol=0.0001, rtol=0.001)

class FCFusePassTRTDynamicDims3Cols1Test(InferencePassTest):

    def setUp(self):
        if False:
            return 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[32, 128, 32], dtype='float32')
            fc_out1 = paddle.static.nn.fc(x=data, size=64, num_flatten_dims=1, activation='relu')
            out = paddle.nn.functional.softmax(fc_out1)
        self.feeds = {'data': np.random.random((32, 128, 32)).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims3Cols1Test.TensorRTParam(1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims3Cols1Test.DynamicShapeParam({'data': [1, 128, 32]}, {'data': [64, 128, 32]}, {'data': [32, 128, 32]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            while True:
                i = 10
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol=0.0001, rtol=0.001)

class FCFusePassTRTDynamicDims3Cols2Test(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[32, 128, 32], dtype='float32')
            fc_out1 = paddle.static.nn.fc(x=data, size=64, num_flatten_dims=2, activation='relu')
            out = paddle.nn.functional.softmax(fc_out1)
        self.feeds = {'data': np.random.random((32, 128, 32)).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims3Cols2Test.TensorRTParam(1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims3Cols2Test.DynamicShapeParam({'data': [1, 32, 32]}, {'data': [64, 256, 32]}, {'data': [32, 128, 32]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            print('Hello World!')
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol=0.0001, rtol=0.001)

class FCFusePassTRTDynamicDims4Cols1Test(InferencePassTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[32, 12, 4, 6], dtype='float32')
            fc_out1 = paddle.static.nn.fc(x=data, size=64, num_flatten_dims=1, activation='relu')
            out = paddle.nn.functional.softmax(fc_out1)
        self.feeds = {'data': np.random.random((32, 12, 4, 6)).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims4Cols1Test.TensorRTParam(1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims4Cols1Test.DynamicShapeParam({'data': [1, 12, 4, 6]}, {'data': [64, 12, 4, 6]}, {'data': [32, 12, 4, 6]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            while True:
                i = 10
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol=0.0001, rtol=0.001)

class FCFusePassTRTDynamicDims4Cols2Test(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[32, 128, 32, 32], dtype='float32')
            fc_out1 = paddle.static.nn.fc(x=data, size=64, num_flatten_dims=2, activation='relu')
            out = paddle.nn.functional.softmax(fc_out1)
        self.feeds = {'data': np.random.random((32, 128, 32, 32)).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims4Cols2Test.TensorRTParam(1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims4Cols2Test.DynamicShapeParam({'data': [1, 64, 32, 32]}, {'data': [64, 256, 32, 32]}, {'data': [32, 128, 32, 32]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol=0.0001, rtol=0.001)

class FCFusePassTRTDynamicDims4Cols3Test(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[32, 128, 32, 32], dtype='float32')
            fc_out1 = paddle.static.nn.fc(x=data, size=64, num_flatten_dims=3, activation='relu')
            out = paddle.nn.functional.softmax(fc_out1)
        self.feeds = {'data': np.random.random((32, 128, 32, 32)).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims4Cols3Test.TensorRTParam(1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims4Cols3Test.DynamicShapeParam({'data': [1, 128, 32, 32]}, {'data': [64, 128, 32, 32]}, {'data': [32, 128, 32, 32]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            return 10
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol=0.0001, rtol=0.001)
if __name__ == '__main__':
    unittest.main()