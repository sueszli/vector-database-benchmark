import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base.core import PassVersionChecker

class TestMKLDNNCpuBfloat16Pass(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_data()
        with base.program_guard(self.main_program, self.startup_program):
            x = paddle.static.data(name='x', shape=[-1] + self.shape_x, dtype=self.d_type)
            out = paddle.transpose(x, perm=[0, 1, 2, 3])
            out = paddle.reshape(out, [0, 0, 0, 0])
            out = paddle.static.nn.fc(out, size=1)
            self.feeds = {'x': np.random.random([self.bs] + self.shape_x).astype(self.d_type)}
            self.fetch_list = [out]

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.bs = 8
        self.d_type = np.float32
        self.shape_x = [12, 10, 1]
        self.shape_y = [12, 1, 64]
        self.enable_mkldnn = True
        self.enable_mkldnn_bfloat16 = True

    def test_check_output(self):
        if False:
            return 10
        use_gpu = False
        self.check_output_with_option(use_gpu, flatten=True)
        self.assertTrue(PassVersionChecker.IsCompatible('cpu_bfloat16_pass'))
if __name__ == '__main__':
    unittest.main()