import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn

class ShuffleChannelFuseTRTPassTest(InferencePassTest):

    def setUp(self):
        if False:
            print('Hello World!')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[-1, 6, 64, 64], dtype='float32')
            reshape1 = paddle.reshape(x=data, shape=[-1, 2, 3, 64, 64])
            trans = paddle.transpose(x=reshape1, perm=[0, 2, 1, 3, 4])
            reshape2 = paddle.reshape(x=trans, shape=[-1, 6, 64, 64])
            out = nn.batch_norm(reshape2, is_test=True)
        self.feeds = {'data': np.random.random([1, 6, 64, 64]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = ShuffleChannelFuseTRTPassTest.TensorRTParam(1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible('shuffle_channel_detect_pass'))
if __name__ == '__main__':
    unittest.main()