import unittest
import numpy as np
from quant_dequant_test import QuantDequantTest
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker

class FCQuantDequantFusePassTRTDims3Cols1Test(QuantDequantTest):

    def setUp(self):
        if False:
            print('Hello World!')

        def network():
            if False:
                return 10
            self.data = paddle.static.data(name='data', shape=[1, 28, 28], dtype='float32')
            self.label = paddle.static.data(name='label', shape=[1, 1], dtype='int64')
            fc_out = paddle.static.nn.fc(x=self.data, size=10, num_flatten_dims=1, bias_attr=False, activation='relu')
            result = F.relu(fc_out)
            loss = paddle.nn.functional.cross_entropy(input=result, label=self.label, reduction='none', use_softmax=False)
            avg_loss = paddle.mean(loss)
            return (avg_loss, result)
        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
        with base.unique_name.guard():
            with base.program_guard(self.main_program, self.startup_program):
                (self.loss, result) = network()
                opt = paddle.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with base.unique_name.guard():
            with base.program_guard(self.test_main_program, self.startup_program):
                network()
        self.feeds = {'data': np.random.random((1, 28, 28)).astype('float32')}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = FCQuantDequantFusePassTRTDims3Cols1Test.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.dynamic_shape_params = FCQuantDequantFusePassTRTDims3Cols1Test.DynamicShapeParam({'data': [1, 28, 28], 'reshape2_1.tmp_0': [1, 1, 10]}, {'data': [2, 28, 28], 'reshape2_1.tmp_0': [2, 1, 10]}, {'data': [1, 28, 28], 'reshape2_1.tmp_0': [1, 1, 10]}, False)
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=0.01, flatten=False, rtol=0.01)
            self.assertTrue(PassVersionChecker.IsCompatible('quant_conv2d_dequant_fuse_pass'))

class FCQuantDequantFusePassTRTDims3Cols2Test(QuantDequantTest):

    def setUp(self):
        if False:
            print('Hello World!')

        def network():
            if False:
                for i in range(10):
                    print('nop')
            self.data = paddle.static.data(name='data', shape=[1, 28, 28], dtype='float32')
            self.label = paddle.static.data(name='label', shape=[1, 1], dtype='int64')
            fc_out = paddle.static.nn.fc(x=self.data, size=28, num_flatten_dims=2, bias_attr=False, activation=None)
            c_out = paddle.reshape(fc_out, shape=[0, 784])
            result = F.relu(c_out)
            loss = paddle.nn.functional.cross_entropy(input=result, label=self.label, reduction='none', use_softmax=False)
            avg_loss = paddle.mean(loss)
            return (avg_loss, result)
        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
        with base.unique_name.guard():
            with base.program_guard(self.main_program, self.startup_program):
                (self.loss, result) = network()
                opt = paddle.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with base.unique_name.guard():
            with base.program_guard(self.test_main_program, self.startup_program):
                network()
        self.feeds = {'data': np.random.random((1, 28, 28)).astype('float32')}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = FCQuantDequantFusePassTRTDims3Cols2Test.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.dynamic_shape_params = FCQuantDequantFusePassTRTDims3Cols2Test.DynamicShapeParam({'data': [1, 28, 28], 'reshape2_0.tmp_0': [1, 784]}, {'data': [4, 28, 28], 'reshape2_0.tmp_0': [4, 784]}, {'data': [1, 28, 28], 'reshape2_0.tmp_0': [1, 784]}, False)
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def test_check_output(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=0.1, flatten=False, rtol=0.1)
            self.assertTrue(PassVersionChecker.IsCompatible('quant_conv2d_dequant_fuse_pass'))

class FCQuantDequantFusePassTRTDims3Cols3Test(QuantDequantTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15

        def network():
            if False:
                for i in range(10):
                    print('nop')
            self.data = paddle.static.data(name='data', shape=[1, 28, 28], dtype='float32')
            self.label = paddle.static.data(name='label', shape=[1, 1], dtype='int64')
            label_shape = paddle.reshape(self.label, shape=[1, 1, 1])
            reshape_out = paddle.reshape(self.data, shape=[1, 14, 14, 4])
            fc_out = paddle.static.nn.fc(x=reshape_out, size=14, num_flatten_dims=3, bias_attr=False, activation=None)
            c_out = paddle.reshape(fc_out, shape=[1, 1, 2744])
            result = F.relu(c_out)
            loss = paddle.nn.functional.cross_entropy(input=result, label=label_shape, reduction='none', use_softmax=False)
            avg_loss = paddle.mean(loss)
            return (avg_loss, result)
        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
        with base.unique_name.guard():
            with base.program_guard(self.main_program, self.startup_program):
                (self.loss, result) = network()
                opt = paddle.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with base.unique_name.guard():
            with base.program_guard(self.test_main_program, self.startup_program):
                network()
        self.feeds = {'data': np.random.random((1, 28, 28)).astype('float32')}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = FCQuantDequantFusePassTRTDims3Cols3Test.TensorRTParam(1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.dynamic_shape_params = FCQuantDequantFusePassTRTDims3Cols3Test.DynamicShapeParam({'data': [1, 28, 28], 'reshape2_1.tmp_0': [1, 14, 14, 4], 'reshape2_2.tmp_0': [1, 1, 2744]}, {'data': [4, 28, 28], 'reshape2_1.tmp_0': [4, 14, 14, 4], 'reshape2_2.tmp_0': [4, 1, 2744]}, {'data': [1, 28, 28], 'reshape2_1.tmp_0': [1, 14, 14, 4], 'reshape2_2.tmp_0': [1, 1, 2744]}, False)
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=1.0, flatten=False, rtol=1.0)
            self.assertTrue(PassVersionChecker.IsCompatible('quant_conv2d_dequant_fuse_pass'))
if __name__ == '__main__':
    unittest.main()