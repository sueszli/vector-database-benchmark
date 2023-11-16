import os
import unittest
import numpy as np
from utils import check_output, extra_cc_args, extra_nvcc_args, paddle_includes
import paddle
import paddle.nn.functional as F
from paddle import static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
file = f'{get_build_directory()}\\custom_linear\\custom_linear.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)
custom_ops = load(name='custom_linear_jit', sources=['custom_linear_op.cc'], extra_include_paths=paddle_includes, extra_cxx_cflags=extra_cc_args, extra_cuda_cflags=extra_nvcc_args, verbose=True)

def linear_dynamic(func, device, dtype, np_x, np_weight, np_bias):
    if False:
        while True:
            i = 10
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)
    weight = paddle.to_tensor(np_weight, dtype=dtype, stop_gradient=False)
    bias = paddle.to_tensor(np_bias, dtype=dtype, stop_gradient=False)
    out = func(x, weight, bias)
    out.backward()
    return (out.numpy(), x.grad.numpy(), weight.grad.numpy(), bias.grad.numpy())

def linear_static(func, device, dtype, np_x, np_weight, np_bias):
    if False:
        for i in range(10):
            print('nop')
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='x', shape=[None, np_x.shape[1]], dtype=dtype)
            weight = static.data(name='weight', shape=np_weight.shape, dtype=dtype)
            bias = static.data(name='bias', shape=np_bias.shape, dtype=dtype)
            x.stop_gradient = False
            weight.stop_gradient = False
            bias.stop_gradient = False
            out = func(x, weight, bias)
            mean_out = paddle.mean(out)
            static.append_backward(mean_out)
            exe = static.Executor()
            exe.run(static.default_startup_program())
            (out_v, x_grad_v, weight_grad_v, bias_grad_v) = exe.run(static.default_main_program(), feed={'x': np_x.astype(dtype), 'weight': np_weight.astype(dtype), 'bias': np_bias.astype(dtype)}, fetch_list=[out.name, x.name + '@GRAD', weight.name + '@GRAD', bias.name + '@GRAD'])
    paddle.disable_static()
    return (out_v, x_grad_v, weight_grad_v, bias_grad_v)

class TestCustomLinearJit(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')
        self.np_x = np.random.random((3, 2)).astype('float32')
        self.np_weight = np.full([2, 4], fill_value=0.5, dtype='float32')
        self.np_bias = np.ones([4], dtype='float32')

    def test_static(self):
        if False:
            i = 10
            return i + 15
        for device in self.devices:
            for dtype in self.dtypes:
                (custom_out, custom_x_grad, custom_weight_grad, custom_bias_grad) = linear_static(custom_ops.custom_linear, device, dtype, self.np_x, self.np_weight, self.np_bias)
                (pd_out, pd_x_grad, pd_weight_grad, pd_bias_grad) = linear_static(F.linear, device, dtype, self.np_x, self.np_weight, self.np_bias)
                check_output(custom_out, pd_out, 'out')
                check_output(custom_x_grad, pd_x_grad, 'x_grad')
                check_output(custom_weight_grad, pd_weight_grad, 'weight_grad')
                check_output(custom_bias_grad, pd_bias_grad, 'bias_grad')

    def test_dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        for device in self.devices:
            for dtype in self.dtypes:
                (custom_out, custom_x_grad, custom_weight_grad, custom_bias_grad) = linear_dynamic(custom_ops.custom_linear, device, dtype, self.np_x, self.np_weight, self.np_bias)
                (pd_out, pd_x_grad, pd_weight_grad, pd_bias_grad) = linear_dynamic(F.linear, device, dtype, self.np_x, self.np_weight, self.np_bias)
                check_output(custom_out, pd_out, 'custom_out')
                check_output(custom_x_grad, pd_x_grad, 'x_grad')
                check_output(custom_weight_grad, pd_weight_grad, 'weight_grad')
                check_output(custom_bias_grad, pd_bias_grad, 'bias_grad')
if __name__ == '__main__':
    unittest.main()