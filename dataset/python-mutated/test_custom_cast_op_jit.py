import os
import unittest
import numpy as np
from utils import extra_cc_args, extra_nvcc_args, paddle_includes, paddle_libraries
import paddle
from paddle import static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
file = f'{get_build_directory()}\\custom_cast_module_jit\\custom_cast_module_jit.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)
custom_module = load(name='custom_cast_module_jit', sources=['custom_cast_op.cc'], extra_include_paths=paddle_includes, extra_library_paths=paddle_libraries, extra_cxx_cflags=extra_cc_args, extra_cuda_cflags=extra_nvcc_args, verbose=True)

def custom_cast_dynamic(device, dtype, np_x):
    if False:
        for i in range(10):
            print('nop')
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype='float32')
    x.stop_gradient = False
    out = custom_module.custom_cast(x, dtype)
    out.stop_gradient = False
    out.backward()
    assert str(out.dtype).split('.')[-1] == dtype
    assert str(x.grad.dtype).split('.')[-1] == dtype

def custom_cast_static(device, dtype, np_x):
    if False:
        return 10
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype='float32')
            x.stop_gradient = False
            out = custom_module.custom_cast(x, dtype)
            static.append_backward(out)
            exe = static.Executor()
            exe.run(static.default_startup_program())
            (out_v, x_grad_v) = exe.run(static.default_main_program(), feed={'X': np_x}, fetch_list=[out.name, x.name + '@GRAD'])
            assert x_grad_v[0].dtype == dtype
            assert out_v[0].dtype == dtype
    paddle.disable_static()
    return out_v

class TestCustomCastOp(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.dtypes = ['float32', 'float64']

    def test_static(self):
        if False:
            while True:
                i = 10
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype('float32')
            custom_cast_static('cpu', dtype, x)

    def test_dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype('float32')
            custom_cast_dynamic('cpu', dtype, x)
if __name__ == '__main__':
    unittest.main()