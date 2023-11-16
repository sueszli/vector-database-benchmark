import unittest
import numpy as np
from pass_test import PassTest
import paddle
from paddle import base
from paddle.base import core

class FCFusePassTest(PassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(name='data', shape=[32, 128], dtype='float32', lod_level=0)
            tmp_0 = paddle.static.nn.fc(x=data, size=128, num_flatten_dims=1, activation='relu')
            tmp_1 = paddle.static.nn.fc(x=tmp_0, size=32, num_flatten_dims=1)
            tmp_2 = paddle.nn.functional.softmax(tmp_1)
        self.feeds = {'data': np.random.random((32, 128)).astype('float32')}
        self.fetch_list = [tmp_0, tmp_1, tmp_2]
        self.pass_names = 'fc_fuse_pass'
        self.fused_op_type = 'fc'
        self.num_fused_ops = 2

    def test_check_output(self):
        if False:
            while True:
                i = 10
        use_gpu_set = [False]
        if core.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            self.pass_attrs = {'fc_fuse_pass': {'use_gpu': use_gpu}}
            place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
            self.check_output_with_place(place, startup_on_cpu=True)
if __name__ == '__main__':
    unittest.main()