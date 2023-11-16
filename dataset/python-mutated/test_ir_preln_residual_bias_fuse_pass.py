import unittest
from pass_test import PassTest
import paddle

class PrelnResidualBiasFusePassTest(PassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with paddle.static.program_guard(self.main_program, self.startup_program):
            x = paddle.static.data(name='x', shape=[128, 768], dtype='float32', lod_level=0)
            bias = paddle.static.create_parameter(shape=[768], dtype='float32')
            y = paddle.static.data(name='y', shape=[128, 768], dtype='float32', lod_level=0)
            x = x + bias
            elementwise_out = x + y
            out = paddle.static.nn.layer_norm(input=elementwise_out)
        self.fetch_list = [out, elementwise_out]
        self.pass_names = 'preln_residual_bias_fuse_pass'
        self.fused_op_type = 'fused_bias_dropout_residual_layer_norm'
        self.num_fused_ops = 1

    def test_check_program(self):
        if False:
            i = 10
            return i + 15
        use_gpu_set = [False]
        if paddle.device.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
            opt_program = self._apply_ir_passes()
            self.check_program(opt_program)

class PrelnResidualBiasFusePassNoBiasTest(PassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        with paddle.static.program_guard(self.main_program, self.startup_program):
            x = paddle.static.data(name='x', shape=[128, 768], dtype='float32', lod_level=0)
            y = paddle.static.data(name='y', shape=[128, 768], dtype='float32', lod_level=0)
            elementwise_out = x + y
            out = paddle.static.nn.layer_norm(input=elementwise_out)
        self.fetch_list = [out, elementwise_out]
        self.pass_names = 'preln_residual_bias_fuse_pass'
        self.fused_op_type = 'fused_bias_dropout_residual_layer_norm'
        self.num_fused_ops = 1

    def test_check_program(self):
        if False:
            while True:
                i = 10
        use_gpu_set = [False]
        if paddle.device.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
            opt_program = self._apply_ir_passes()
            self.check_program(opt_program)
if __name__ == '__main__':
    unittest.main()