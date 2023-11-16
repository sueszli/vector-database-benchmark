import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig

class TransposeFlattenConcatFusePassTRTTest(InferencePassTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(name='data1', shape=[8, 32, 128], dtype='float32')
            data2 = paddle.static.data(name='data2', shape=[8, 32, 128], dtype='float32')
            trans1 = paddle.transpose(data1, perm=[0, 2, 1])
            trans2 = paddle.transpose(data2, perm=[0, 2, 1])
            flatt1 = paddle.flatten(trans1, 1, -1)
            flatt2 = paddle.flatten(trans2, 1, -1)
            concat_out = paddle.concat([flatt1, flatt2], axis=1)
            reshape_out = paddle.reshape(concat_out, [-1, 0, 1, 1])
            out = paddle.static.nn.batch_norm(reshape_out, is_test=True)
        self.feeds = {'data1': np.random.random([8, 32, 128]).astype('float32'), 'data2': np.random.random([8, 32, 128]).astype('float32')}
        self.enable_trt = True
        self.trt_parameters = TransposeFlattenConcatFusePassTRTTest.TensorRTParam(1 << 20, 8, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
if __name__ == '__main__':
    unittest.main()