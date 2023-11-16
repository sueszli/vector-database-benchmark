import os
import unittest
import numpy as np
from utils import check_output, extra_cc_args, extra_nvcc_args, paddle_includes
import paddle
from paddle import static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
file = f'{get_build_directory()}\\custom_conj\\custom_conj.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)
custom_ops = load(name='custom_conj_jit', sources=['custom_conj_op.cc'], extra_include_paths=paddle_includes, extra_cxx_cflags=extra_cc_args, extra_cuda_cflags=extra_nvcc_args, verbose=True)

def is_complex(dtype):
    if False:
        for i in range(10):
            print('nop')
    return dtype == paddle.base.core.VarDesc.VarType.COMPLEX64 or dtype == paddle.base.core.VarDesc.VarType.COMPLEX128

def to_complex(dtype):
    if False:
        return 10
    if dtype == 'float32':
        return np.complex64
    elif dtype == 'float64':
        return np.complex128
    else:
        return dtype

def conj_dynamic(func, dtype, np_input):
    if False:
        for i in range(10):
            print('nop')
    paddle.set_device('cpu')
    x = paddle.to_tensor(np_input)
    out = func(x)
    out.stop_gradient = False
    sum_out = paddle.sum(out)
    if is_complex(sum_out.dtype):
        sum_out.real().backward()
    else:
        sum_out.backward()
    if x.grad is None:
        return (out.numpy(), x.grad)
    else:
        return (out.numpy(), x.grad.numpy())

def conj_static(func, shape, dtype, np_input):
    if False:
        return 10
    paddle.enable_static()
    paddle.set_device('cpu')
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='x', shape=shape, dtype=dtype)
            x.stop_gradient = False
            out = func(x)
            sum_out = paddle.sum(out)
            static.append_backward(sum_out)
            exe = static.Executor()
            exe.run(static.default_startup_program())
            (out_v, x_grad_v) = exe.run(static.default_main_program(), feed={'x': np_input}, fetch_list=[out.name, x.name + '@GRAD'])
    paddle.disable_static()
    return (out_v, x_grad_v)

class TestCustomConjJit(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.dtypes = ['float32', 'float64']
        self.shape = [2, 20, 2, 3]

    def test_dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype)
            (out, x_grad) = conj_dynamic(custom_ops.custom_conj, dtype, np_input)
            (pd_out, pd_x_grad) = conj_dynamic(paddle.conj, dtype, np_input)
            check_output(out, pd_out, 'out')
            check_output(x_grad, pd_x_grad, "x's grad")

    def test_static(self):
        if False:
            print('Hello World!')
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype)
            (out, x_grad) = conj_static(custom_ops.custom_conj, self.shape, dtype, np_input)
            (pd_out, pd_x_grad) = conj_static(paddle.conj, self.shape, dtype, np_input)
            check_output(out, pd_out, 'out')
            check_output(x_grad, pd_x_grad, "x's grad")

    def test_complex_dynamic(self):
        if False:
            return 10
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype) + 1j * np.random.random(self.shape).astype(dtype)
            (out, x_grad) = conj_dynamic(custom_ops.custom_conj, to_complex(dtype), np_input)
            (pd_out, pd_x_grad) = conj_dynamic(paddle.conj, to_complex(dtype), np_input)
            check_output(out, pd_out, 'out')
            check_output(x_grad, pd_x_grad, "x's grad")
if __name__ == '__main__':
    unittest.main()