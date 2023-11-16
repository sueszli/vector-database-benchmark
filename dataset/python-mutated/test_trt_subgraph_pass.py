import os
import shutil
import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn

class TensorRTSubgraphPassFcTest(InferencePassTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, 64, 64], dtype='float32')
            fc_out = paddle.static.nn.fc(x=[data], activation=None, size=1000)
            reshape_out = paddle.reshape(x=fc_out, shape=[1, 1000])
        self.feeds = {'data': np.random.random([1, 6, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassFcTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [reshape_out]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassConcatTest(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(name='data1', shape=[-1, 3, 64, 64], dtype='float32')
            data2 = paddle.static.data(name='data2', shape=[-1, 3, 64, 64], dtype='float32')
            concat_out = paddle.concat([data1, data2], axis=2)
            out = nn.batch_norm(concat_out, is_test=True)
        self.feeds = {'data1': np.random.random([1, 3, 64, 64]).astype('float32'), 'data2': np.random.random([1, 3, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConcatTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            return 10
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassSplitTest(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 64, 64], dtype='float32')
            split_out = paddle.split(data, axis=-1, num_or_sections=2)
            out = nn.batch_norm(split_out[0], is_test=True)
        self.feeds = {'data': np.random.random([1, 3, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassSplitTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassSplitSerializeTest(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 64, 64], dtype='float32')
            split_out = paddle.split(data, axis=-1, num_or_sections=2)
            out = nn.batch_norm(split_out[0], is_test=True)
        self.feeds = {'data': np.random.random([1, 3, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassSplitTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            return 10
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            opt_path = os.path.join(self.path, '_opt_cache')
            if os.path.exists(opt_path):
                shutil.rmtree(opt_path)
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassDynamicSplitFp16SerializeTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 64, 64], dtype='float32')
            split_out = paddle.split(data, axis=-1, num_or_sections=2)
            out = nn.batch_norm(split_out[0], is_test=True)
        self.feeds = {'data': np.random.random([1, 3, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassSplitTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False)
        self.dynamic_shape_params = TensorRTSubgraphPassDynamicSplitFp16SerializeTest.DynamicShapeParam({'data': [1, 3, 8, 64]}, {'data': [1, 3, 512, 64]}, {'data': [1, 3, 256, 64]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            opt_path = os.path.join(self.path, '_opt_cache')
            if os.path.exists(opt_path):
                shutil.rmtree(opt_path)
            self.check_output_with_option(use_gpu, 0.001)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassInstanceNormTest(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 64, 64], dtype='float32')
            param_attr = base.ParamAttr(name='instance_norm_w', initializer=paddle.nn.initializer.Constant(value=1.0))
            bias_attr = base.ParamAttr(name='instance_norm_b', initializer=paddle.nn.initializer.Constant(value=0.0))
            out = paddle.static.nn.instance_norm(input=data, param_attr=param_attr, bias_attr=bias_attr)
        self.feeds = {'data': np.random.random([1, 3, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassInstanceNormTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=0.0001, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassTransposeTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, 64, 64], dtype='float32')
            transpose_out = self.append_transpose(data)
            out = nn.batch_norm(transpose_out, is_test=True)
        self.feeds = {'data': np.random.random([1, 6, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassTransposeTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_transpose(self, data):
        if False:
            for i in range(10):
                print('nop')
        return paddle.transpose(data, [0, 3, 1, 2])

    def test_check_output(self):
        if False:
            return 10
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassLayerNormTest(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 64, 64], dtype='float32')
            out = paddle.static.nn.layer_norm(data, begin_norm_axis=self.begin_norm_axis)
        self.feeds = {'data': np.random.random([1, 3, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassLayerNormTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        if False:
            return 10
        self.begin_norm_axis = 1

    def test_check_output(self):
        if False:
            return 10
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassLayerNormDynamicTest(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 64, 64], dtype='float32')
            out = paddle.static.nn.layer_norm(data, begin_norm_axis=self.begin_norm_axis)
        self.feeds = {'data': np.random.random([1, 3, 64, 64]).astype('float32')}
        self.set_trt_params()
        self.fetch_list = [out]

    def set_trt_params(self):
        if False:
            print('Hello World!')
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassLayerNormDynamicTest.TensorRTParam(1 << 30, 32, 0, self.precision, self.serialize, False)
        self.dynamic_shape_params = TensorRTSubgraphPassLayerNormDynamicTest.DynamicShapeParam({'data': [1, 3, 64, 64]}, {'data': [8, 8, 64, 64]}, {'data': [4, 4, 64, 64]}, False)

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.begin_norm_axis = 2
        self.precision = AnalysisConfig.Precision.Float32
        self.serialize = True

    def test_check_output(self):
        if False:
            while True:
                i = 10
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassLayerNormDynamicFP16Test(TensorRTSubgraphPassLayerNormDynamicTest):

    def set_params(self):
        if False:
            print('Hello World!')
        self.begin_norm_axis = 2
        self.precision = AnalysisConfig.Precision.Half
        self.serialize = True

    def test_check_output(self):
        if False:
            while True:
                i = 10
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=0.01, rtol=0.01)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassLayerNormBeginNormAxis2Test(TensorRTSubgraphPassLayerNormTest):

    def set_params(self):
        if False:
            return 10
        self.begin_norm_axis = 2

class TensorRTSubgraphPassLayerNormBeginNormAxis3Test(TensorRTSubgraphPassLayerNormTest):

    def set_params(self):
        if False:
            while True:
                i = 10
        self.begin_norm_axis = 3

class TensorRTSubgraphPassElementwiseTest(InferencePassTest):

    def setUp(self):
        if False:
            return 10
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(name='data1', shape=[-1, 3, 64, 64], dtype='float32')
            data2 = paddle.static.data(name='data2', shape=[-1, 3, 64, 64], dtype='float32')
            eltwise_out = self.append_eltwise(data1, data2)
            out = nn.batch_norm(eltwise_out, is_test=True)
        self.feeds = {'data1': np.random.random([1, 3, 64, 64]).astype('float32'), 'data2': np.random.random([1, 3, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassElementwiseTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        if False:
            i = 10
            return i + 15
        return paddle.add(x=data1, y=data2)

    def test_check_output(self):
        if False:
            print('Hello World!')
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TensorRTSubgraphPassElementwiseMulTest(TensorRTSubgraphPassElementwiseTest):

    def append_eltwise(self, data1, data2):
        if False:
            while True:
                i = 10
        return paddle.multiply(x=data1, y=data2)

class TensorRTSubgraphPassElementwiseSerializeTest(TensorRTSubgraphPassElementwiseTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.trt_parameters = TensorRTSubgraphPassElementwiseTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False)

    def test_check_output(self):
        if False:
            return 10
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        super().test_check_output()

class TensorRTSubgraphPassElementwiseBroadcastDynamicTest(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(name='data1', shape=[-1, 3, 64, 64], dtype='float32')
            data2 = paddle.static.data(name='data2', shape=[64, 64], dtype='float32')
            eltwise_out = self.append_eltwise(data1, data2)
            out = nn.batch_norm(eltwise_out, is_test=True)
        self.feeds = {'data1': np.random.random([1, 3, 64, 64]).astype('float32'), 'data2': np.random.random([64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassElementwiseBroadcastDynamicTest.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False)
        self.dynamic_shape_params = TensorRTSubgraphPassElementwiseBroadcastDynamicTest.DynamicShapeParam({'data1': [1, 3, 8, 64], 'data2': [8, 64]}, {'data1': [1, 3, 512, 64], 'data2': [512, 64]}, {'data1': [1, 3, 256, 64], 'data2': [256, 64]}, False)
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        if False:
            while True:
                i = 10
        return paddle.add(x=data1, y=data2)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
if __name__ == '__main__':
    unittest.main()