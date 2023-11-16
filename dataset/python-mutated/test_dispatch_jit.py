import os
import unittest
import numpy as np
from utils import extra_cc_args, paddle_includes
import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
file = f'{get_build_directory()}\\dispatch_op\\dispatch_op.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)
dispatch_op = load(name='dispatch_op', sources=['dispatch_test_op.cc'], extra_include_paths=paddle_includes, extra_cxx_cflags=extra_cc_args, verbose=True)

class TestJitDispatch(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.set_device('cpu')

    def run_dispatch_test(self, func, dtype):
        if False:
            for i in range(10):
                print('nop')
        np_x = np.ones([2, 2]).astype(dtype)
        x = paddle.to_tensor(np_x)
        out = func(x)
        np_x = x.numpy()
        np_out = out.numpy()
        self.assertTrue(dtype in str(np_out.dtype))
        np.testing.assert_array_equal(np_x, np_out, err_msg=f'custom op x: {np_x},\n custom op out: {np_out}')

    def test_dispatch_integer(self):
        if False:
            i = 10
            return i + 15
        dtypes = ['int32', 'int64', 'int8', 'uint8', 'int16']
        for dtype in dtypes:
            self.run_dispatch_test(dispatch_op.dispatch_test_integer, dtype)

    def test_dispatch_complex(self):
        if False:
            i = 10
            return i + 15
        dtypes = ['complex64', 'complex128']
        for dtype in dtypes:
            self.run_dispatch_test(dispatch_op.dispatch_test_complex, dtype)

    def test_dispatch_float_and_integer(self):
        if False:
            while True:
                i = 10
        dtypes = ['float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'int16']
        for dtype in dtypes:
            self.run_dispatch_test(dispatch_op.dispatch_test_float_and_integer, dtype)

    def test_dispatch_float_and_complex(self):
        if False:
            for i in range(10):
                print('nop')
        dtypes = ['float32', 'float64', 'complex64', 'complex128']
        for dtype in dtypes:
            self.run_dispatch_test(dispatch_op.dispatch_test_float_and_complex, dtype)

    def test_dispatch_float_and_integer_and_complex(self):
        if False:
            return 10
        dtypes = ['float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'int16', 'complex64', 'complex128']
        for dtype in dtypes:
            self.run_dispatch_test(dispatch_op.dispatch_test_float_and_integer_and_complex, dtype)

    def test_dispatch_float_and_half(self):
        if False:
            while True:
                i = 10
        dtypes = ['float32', 'float64', 'float16']
        for dtype in dtypes:
            self.run_dispatch_test(dispatch_op.dispatch_test_float_and_half, dtype)
if __name__ == '__main__':
    unittest.main()