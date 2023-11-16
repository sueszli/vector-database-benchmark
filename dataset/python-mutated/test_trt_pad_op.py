import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig
from paddle.static import nn

class PadOpTRTTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[1, 3, 128, 128], dtype='float32')
            pad_out = paddle.nn.functional.pad(x=data, pad=[0, 0, 0, 0, 0, 1, 1, 2], value=0.0)
            out = nn.batch_norm(pad_out, is_test=True)
        self.feeds = {'data': np.random.random((1, 3, 128, 128)).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = PadOpTRTTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            while True:
                i = 10
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])
if __name__ == '__main__':
    unittest.main()