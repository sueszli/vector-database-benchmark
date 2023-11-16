import os
import unittest
import numpy as np
from utils import extra_cc_args, extra_nvcc_args, paddle_includes
import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
file = f'{get_build_directory()}\\custom_attrs_jit\\custom_attrs_jit.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)
custom_attrs = load(name='custom_attrs_jit', sources=['attr_test_op.cc'], extra_include_paths=paddle_includes, extra_cxx_cflags=extra_cc_args, extra_cuda_cflags=extra_nvcc_args, verbose=True)

class TestJitCustomAttrs(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.set_device('cpu')
        self.bool_attr = True
        self.int_attr = 10
        self.float_attr = 3.14
        self.int64_attr = 10000000000
        self.str_attr = 'StrAttr'
        self.int_vec_attr = [10, 10, 10]
        self.float_vec_attr = [3.14, 3.14, 3.14]
        self.int64_vec_attr = [10000000000, 10000000000, 10000000000]
        self.str_vec_attr = ['StrAttr', 'StrAttr', 'StrAttr']

    def test_func_attr_value(self):
        if False:
            print('Hello World!')
        x = paddle.ones([2, 2], dtype='float32')
        x.stop_gradient = False
        out = custom_attrs.attr_test(x, self.bool_attr, self.int_attr, self.float_attr, self.int64_attr, self.str_attr, self.int_vec_attr, self.float_vec_attr, self.int64_vec_attr, self.str_vec_attr)
        out.stop_gradient = False
        out.backward()
        np.testing.assert_array_equal(x.numpy(), out.numpy())

    def test_const_attr_value(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.ones([2, 2], dtype='float32')
        x.stop_gradient = False
        out = custom_attrs.const_attr_test(x, self.bool_attr, self.int_attr, self.float_attr, self.int64_attr, self.str_attr, self.int_vec_attr, self.float_vec_attr, self.int64_vec_attr, self.str_vec_attr)
        out.stop_gradient = False
        out.backward()
        np.testing.assert_array_equal(x.numpy(), out.numpy())
if __name__ == '__main__':
    unittest.main()