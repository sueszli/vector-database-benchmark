import os
import unittest
import numpy as np
from test_custom_relu_op_setup import custom_relu_dynamic, custom_relu_static
from utils import IS_MAC, extra_cc_args, extra_nvcc_args, paddle_includes, paddle_libraries
import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
file = f'{get_build_directory()}\\custom_relu_module_jit\\custom_relu_module_jit.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)
sources = ['custom_relu_op.cc', 'custom_relu_op_dup.cc']
if not IS_MAC:
    sources.append('custom_relu_op.cu')
custom_module = load(name='custom_relu_module_jit', sources=sources, extra_include_paths=paddle_includes, extra_library_paths=paddle_libraries, extra_cxx_cflags=extra_cc_args, extra_cuda_cflags=extra_nvcc_args, verbose=True)

class TestJITLoad(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.custom_ops = [custom_module.custom_relu, custom_module.custom_relu_dup, custom_module.custom_relu_no_x_in_backward, custom_module.custom_relu_out]
        self.dtypes = ['float32', 'float64']
        if paddle.is_compiled_with_cuda():
            self.dtypes.append('float16')
        self.devices = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')

    def test_static(self):
        if False:
            for i in range(10):
                print('nop')
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out = custom_relu_static(custom_op, device, dtype, x)
                    pd_out = custom_relu_static(custom_op, device, dtype, x, False)
                    np.testing.assert_array_equal(out, pd_out, err_msg=f'custom op out: {out},\n paddle api out: {pd_out}')

    def test_dynamic(self):
        if False:
            i = 10
            return i + 15
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    (out, x_grad) = custom_relu_dynamic(custom_op, device, dtype, x)
                    (pd_out, pd_x_grad) = custom_relu_dynamic(custom_op, device, dtype, x, False)
                    np.testing.assert_array_equal(out, pd_out, err_msg=f'custom op out: {out},\n paddle api out: {pd_out}')
                    np.testing.assert_array_equal(x_grad, pd_x_grad, err_msg='custom op x grad: {},\n paddle api x grad: {}'.format(x_grad, pd_x_grad))

    def test_exception(self):
        if False:
            return 10
        caught_exception = False
        try:
            x = np.random.uniform(-1, 1, [4, 8]).astype('int32')
            custom_relu_dynamic(custom_module.custom_relu, 'cpu', 'int32', x)
        except OSError as e:
            caught_exception = True
            self.assertTrue('relu_cpu_forward' in str(e))
            self.assertTrue('int32' in str(e))
            self.assertTrue('custom_relu_op.cc' in str(e))
        self.assertTrue(caught_exception)
        caught_exception = False
        if IS_MAC:
            return
        try:
            x = np.random.uniform(-1, 1, [4, 8]).astype('int32')
            custom_relu_dynamic(custom_module.custom_relu, 'gpu', 'int32', x)
        except OSError as e:
            caught_exception = True
            self.assertTrue('relu_cuda_forward_kernel' in str(e))
            self.assertTrue('int32' in str(e))
            self.assertTrue('custom_relu_op.cu' in str(e))
        self.assertTrue(caught_exception)

    def test_load_multiple_module(self):
        if False:
            return 10
        custom_module = load(name='custom_conj_jit', sources=['custom_conj_op.cc'], extra_include_paths=paddle_includes, extra_cxx_cflags=extra_cc_args, extra_cuda_cflags=extra_nvcc_args, verbose=True)
        custom_conj = custom_module.custom_conj
        self.assertIsNotNone(custom_conj)
if __name__ == '__main__':
    unittest.main()