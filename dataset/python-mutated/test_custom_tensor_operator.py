import os
import unittest
import numpy as np
from utils import check_output, check_output_allclose, extra_cc_args, paddle_includes
import paddle
from paddle import static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
file = f'{get_build_directory()}\\custom_tensor_operator\\custom_tensor_operator.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

def test_custom_add_dynamic(func, device, dtype, np_x, use_func=True):
    if False:
        print('Hello World!')
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype)
    x.stop_gradient = False
    if use_func:
        out = func(x)
    else:
        out = x + 1
    out.stop_gradient = False
    out.backward()
    if x.grad is None:
        return (out.numpy(), x.grad)
    else:
        return (out.numpy(), x.grad.numpy())

def test_custom_add_static(func, device, dtype, np_x, use_func=True):
    if False:
        i = 10
        return i + 15
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            if use_func:
                out = func(x)
            else:
                out = x + 1
            static.append_backward(out)
            exe = static.Executor()
            exe.run(static.default_startup_program())
            out_v = exe.run(static.default_main_program(), feed={'X': np_x}, fetch_list=[out.name])
    paddle.disable_static()
    return out_v

def test_custom_subtract_dynamic(func, device, dtype, np_x, use_func=True):
    if False:
        while True:
            i = 10
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype)
    x.stop_gradient = False
    if use_func:
        out = func(x)
    else:
        out = x - 1
    out.stop_gradient = False
    out.backward()
    if x.grad is None:
        return (out.numpy(), x.grad)
    else:
        return (out.numpy(), x.grad.numpy())

def test_custom_subtract_static(func, device, dtype, np_x, use_func=True):
    if False:
        for i in range(10):
            print('nop')
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            if use_func:
                out = func(x)
            else:
                out = x - 1
            static.append_backward(out)
            exe = static.Executor()
            exe.run(static.default_startup_program())
            out_v = exe.run(static.default_main_program(), feed={'X': np_x}, fetch_list=[out.name])
    paddle.disable_static()
    return out_v

def test_custom_multiply_dynamic(func, device, dtype, np_x, use_func=True):
    if False:
        while True:
            i = 10
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype)
    x.stop_gradient = False
    if use_func:
        out = func(x)
    else:
        out = x * 5
    out.stop_gradient = False
    out.backward()
    if x.grad is None:
        return (out.numpy(), x.grad)
    else:
        return (out.numpy(), x.grad.numpy())

def test_custom_multiply_static(func, device, dtype, np_x, use_func=True):
    if False:
        i = 10
        return i + 15
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            if use_func:
                out = func(x)
            else:
                out = x * 5
            static.append_backward(out)
            exe = static.Executor()
            exe.run(static.default_startup_program())
            out_v = exe.run(static.default_main_program(), feed={'X': np_x}, fetch_list=[out.name])
    paddle.disable_static()
    return out_v

def test_custom_divide_dynamic(func, device, dtype, np_x, use_func=True):
    if False:
        print('Hello World!')
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype)
    x.stop_gradient = False
    if use_func:
        out = func(x)
    else:
        out = paddle.reciprocal(x)
    out.stop_gradient = False
    out.backward()
    if x.grad is None:
        return (out.numpy(), x.grad)
    else:
        return (out.numpy(), x.grad.numpy())

def test_custom_divide_static(func, device, dtype, np_x, use_func=True):
    if False:
        while True:
            i = 10
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[4, 8], dtype=dtype)
            x.stop_gradient = False
            if use_func:
                out = func(x)
            else:
                out = paddle.reciprocal(x)
            static.append_backward(out)
            exe = static.Executor()
            exe.run(static.default_startup_program())
            out_v = exe.run(static.default_main_program(), feed={'X': np_x}, fetch_list=[out.name])
    paddle.disable_static()
    return out_v

class TestJITLoad(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.custom_module = load(name='custom_tensor_operator', sources=['custom_tensor_operator.cc'], extra_include_paths=paddle_includes, extra_cxx_cflags=extra_cc_args, verbose=True)
        self.devices = ['cpu']
        self.dtypes = ['float32', 'float64']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')
            self.dtypes.append('float16')

    def test_all(self):
        if False:
            return 10
        self.add = self.custom_module.custom_add
        self.subtract = self.custom_module.custom_subtract
        self.multiply = self.custom_module.custom_multiply
        self.divide = self.custom_module.custom_divide
        self._test_static()
        self._test_dynamic()
        self.add = self.custom_module.custom_scalar_add
        self.subtract = self.custom_module.custom_scalar_subtract
        self.multiply = self.custom_module.custom_scalar_multiply
        self.divide = self.custom_module.custom_scalar_divide
        self._test_static()
        self._test_dynamic()
        self.add = self.custom_module.custom_left_scalar_add
        self.subtract = self.custom_module.custom_left_scalar_subtract
        self.multiply = self.custom_module.custom_left_scalar_multiply
        self.divide = self.custom_module.custom_left_scalar_divide
        self._test_static()
        self._test_dynamic()
        self._test_logical_operants()
        self._test_compare_operants()

    def _test_static(self):
        if False:
            while True:
                i = 10
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out = test_custom_add_static(self.add, device, dtype, x)
                pd_out = test_custom_add_static(self.add, device, dtype, x, False)
                check_output_allclose(out, pd_out, 'out', rtol=1e-05, atol=1e-08)
                out = test_custom_subtract_static(self.subtract, device, dtype, x)
                pd_out = test_custom_subtract_static(self.subtract, device, dtype, x, False)
                check_output_allclose(out, pd_out, 'out', rtol=1e-05, atol=1e-08)
                out = test_custom_multiply_static(self.multiply, device, dtype, x)
                pd_out = test_custom_multiply_static(self.multiply, device, dtype, x, False)
                check_output_allclose(out, pd_out, 'out', rtol=1e-05, atol=1e-08)
                out = test_custom_divide_static(self.divide, device, dtype, x)
                pd_out = test_custom_divide_static(self.divide, device, dtype, x, False)
                check_output_allclose(out, pd_out, 'out', rtol=1e-05, atol=1e-08)

    def _test_dynamic(self):
        if False:
            i = 10
            return i + 15
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                (out, x_grad) = test_custom_add_dynamic(self.add, device, dtype, x)
                (pd_out, pd_x_grad) = test_custom_add_dynamic(self.add, device, dtype, x, False)
                check_output_allclose(out, pd_out, 'out', rtol=1e-05, atol=1e-08)
                check_output_allclose(x_grad, pd_x_grad, 'x_grad', rtol=1e-05, atol=1e-08)
                (out, x_grad) = test_custom_subtract_dynamic(self.subtract, device, dtype, x)
                (pd_out, pd_x_grad) = test_custom_subtract_dynamic(self.subtract, device, dtype, x, False)
                check_output_allclose(out, pd_out, 'out', rtol=1e-05, atol=1e-08)
                check_output_allclose(x_grad, pd_x_grad, 'x_grad', rtol=1e-05, atol=1e-08)
                (out, x_grad) = test_custom_multiply_dynamic(self.multiply, device, dtype, x)
                (pd_out, pd_x_grad) = test_custom_multiply_dynamic(self.multiply, device, dtype, x, False)
                check_output_allclose(out, pd_out, 'out', rtol=1e-05, atol=1e-08)
                check_output_allclose(x_grad, pd_x_grad, 'x_grad', rtol=1e-05, atol=1e-08)
                (out, x_grad) = test_custom_divide_dynamic(self.divide, device, dtype, x)
                (pd_out, pd_x_grad) = test_custom_divide_dynamic(self.divide, device, dtype, x, False)
                check_output_allclose(out, pd_out, 'out', rtol=1e-05, atol=1e-08)

    def _test_logical_operants(self):
        if False:
            i = 10
            return i + 15
        for device in self.devices:
            paddle.set_device(device)
            np_x = paddle.randint(0, 2, [4, 8])
            x = paddle.to_tensor(np_x, dtype='int32')
            np_y = paddle.randint(0, 2, [4, 8])
            y = paddle.to_tensor(np_y, dtype='int32')
            out = self.custom_module.custom_logical_and(x, y)
            pd_out = paddle.bitwise_and(x, y)
            check_output(out.numpy(), pd_out.numpy(), 'out')
            out = self.custom_module.custom_logical_or(x, y)
            pd_out = paddle.bitwise_or(x, y)
            check_output(out.numpy(), pd_out.numpy(), 'out')
            out = self.custom_module.custom_logical_xor(x, y)
            pd_out = paddle.bitwise_xor(x, y)
            check_output(out.numpy(), pd_out.numpy(), 'out')
            out = self.custom_module.custom_logical_not(x)
            pd_out = paddle.bitwise_not(x)
            check_output(out.numpy(), pd_out.numpy(), 'out')

    def _test_compare_operants(self):
        if False:
            return 10
        for device in self.devices:
            paddle.set_device(device)
            np_x = paddle.randint(0, 2, [4, 8])
            x = paddle.to_tensor(np_x, dtype='int32')
            np_y = paddle.randint(0, 2, [4, 8])
            y = paddle.to_tensor(np_y, dtype='int32')
            out = self.custom_module.custom_less_than(x, y)
            pd_out = paddle.less_than(x, y)
            check_output(out.numpy(), pd_out.numpy(), 'out')
            out = self.custom_module.custom_less_equal(x, y)
            pd_out = paddle.less_equal(x, y)
            check_output(out.numpy(), pd_out.numpy(), 'out')
            out = self.custom_module.custom_equal(x, y)
            pd_out = paddle.equal(x, y)
            check_output(out.numpy(), pd_out.numpy(), 'out')
            out = self.custom_module.custom_not_equal(x, y)
            pd_out = paddle.not_equal(x, y)
            check_output(out.numpy(), pd_out.numpy(), 'out')
            out = self.custom_module.custom_greater_than(x, y)
            pd_out = paddle.greater_than(x, y)
            check_output(out.numpy(), pd_out.numpy(), 'out')
            out = self.custom_module.custom_greater_equal(x, y)
            pd_out = paddle.greater_equal(x, y)
            check_output(out.numpy(), pd_out.numpy(), 'out')
if __name__ == '__main__':
    unittest.main()