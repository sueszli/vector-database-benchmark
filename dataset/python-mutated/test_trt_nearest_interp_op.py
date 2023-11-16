import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn

class TRTNearestInterpTest(InferencePassTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            if self.data_layout == 'NCHW':
                shape = [-1, self.channels, self.origin_shape[0], self.origin_shape[1]]
            else:
                shape = [-1, self.origin_shape[0], self.origin_shape[1], self.channels]
            data = paddle.static.data(name='data', shape=shape, dtype='float32')
            resize_out = self.append_nearest_interp(data)
            out = nn.batch_norm(resize_out, is_test=True)
        if self.data_layout == 'NCHW':
            shape = [self.bs, self.channels, self.origin_shape[0], self.origin_shape[1]]
        else:
            shape = [self.bs, self.origin_shape[0], self.origin_shape[1], self.channels]
        self.feeds = {'data': np.random.random(shape).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TRTNearestInterpTest.TensorRTParam(1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.bs = 4
        self.scale = 0
        self.channels = 3
        self.origin_shape = (4, 4)
        self.resize_shape = (16, 16)
        self.align_corners = True
        self.data_layout = 'NCHW'

    def append_nearest_interp(self, data):
        if False:
            print('Hello World!')
        if self.scale > 0.0:
            return paddle.nn.functional.interpolate(data, scale_factor=self.scale, data_format=self.data_layout)
        return paddle.nn.functional.interpolate(data, size=self.resize_shape, data_format=self.data_layout)

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

class TRTNearestInterpTest1(TRTNearestInterpTest):

    def set_params(self):
        if False:
            for i in range(10):
                print('nop')
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)
        self.resize_shape = (32, 32)
        self.align_corners = True
        self.data_layout = 'NCHW'

class TRTNearestInterpTest2(TRTNearestInterpTest):

    def set_params(self):
        if False:
            return 10
        self.bs = 4
        self.scale = 2.0
        self.channels = 3
        self.origin_shape = (16, 16)
        self.resize_shape = (32, 32)
        self.align_corners = False
        self.data_layout = 'NCHW'

class TRTNearestInterpTest3(TRTNearestInterpTest):

    def set_params(self):
        if False:
            print('Hello World!')
        self.bs = 4
        self.scale = 0
        self.channels = 3
        self.origin_shape = (16, 16)
        self.resize_shape = (32, 32)
        self.align_corners = False
        self.data_layout = 'NCHW'

class TRTNearestInterpTest4(TRTNearestInterpTest):

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)
        self.resize_shape = (47, 12)
        self.align_corners = False
        self.data_layout = 'NCHW'

class TRTNearestInterpTest5(TRTNearestInterpTest):

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)
        self.resize_shape = (32, 32)
        self.align_corners = True
        self.data_layout = 'NHWC'

class TRTNearestInterpTest6(TRTNearestInterpTest):

    def set_params(self):
        if False:
            i = 10
            return i + 15
        self.bs = 4
        self.scale = 2.0
        self.channels = 3
        self.origin_shape = (16, 16)
        self.resize_shape = (32, 32)
        self.align_corners = False
        self.data_layout = 'NHWC'

class TRTNearestInterpTest7(TRTNearestInterpTest):

    def set_params(self):
        if False:
            print('Hello World!')
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)
        self.resize_shape = (32, 32)
        self.align_corners = False
        self.data_layout = 'NHWC'

class TRTNearestInterpTest8(TRTNearestInterpTest):

    def set_params(self):
        if False:
            for i in range(10):
                print('nop')
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)
        self.resize_shape = (47, 12)
        self.align_corners = False
        self.data_layout = 'NHWC'

class TRTNearestInterpTest9(TRTNearestInterpTest):

    def set_params(self):
        if False:
            while True:
                i = 10
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)
        self.resize_shape = (47, 12)
        self.align_corners = False
        self.data_layout = 'NHWC'
if __name__ == '__main__':
    unittest.main()