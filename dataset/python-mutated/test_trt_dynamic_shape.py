import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig

class TRTDynamicShapeTest(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 3, 16, 16], dtype='float32')
            out = paddle.static.nn.conv2d(input=data, num_filters=3, filter_size=3, groups=1, padding=[1, 1], bias_attr=False, act=None)
        self.feeds = self.set_feeds()
        self.enable_trt = True
        self.trt_parameters = TRTDynamicShapeTest.TensorRTParam(1 << 30, 1, 1, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = TRTDynamicShapeTest.DynamicShapeParam({'data': [1, 3, 8, 8]}, {'data': [1, 3, 32, 32]}, {'data': [1, 3, 16, 16]}, False)
        self.fetch_list = [out]

    def set_feeds(self):
        if False:
            print('Hello World!')
        return {'data': np.random.random([1, 3, 16, 16]).astype('float32')}

    def test_check_output(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)

class TRTDynamicShapeOutOfBound1Test(TRTDynamicShapeTest):

    def set_feeds(self):
        if False:
            print('Hello World!')
        return {'data': np.random.random([1, 3, 64, 16]).astype('float32')}

    def test_check_output(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            use_gpu = True
            with self.assertRaisesRegex(ValueError, "The fed Variable 'data' should have dimensions"):
                self.check_output_with_option(use_gpu)

class TRTDynamicShapeOutOfBound3Test(TRTDynamicShapeTest):

    def set_feeds(self):
        if False:
            print('Hello World!')
        return {'data': np.random.random([1, 3, 4, 16]).astype('float32')}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            with self.assertRaisesRegex(ValueError, "The fed Variable 'data' should have dimensions"):
                self.check_output_with_option(use_gpu)
if __name__ == '__main__':
    unittest.main()