import os
import unittest
import numpy as np
from utils import check_output, extra_cc_args, extra_nvcc_args, paddle_includes
import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
file = f'{get_build_directory()}\\custom_simple_slice\\custom_simple_slice.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)
custom_ops = load(name='custom_simple_slice_jit', sources=['custom_simple_slice_op.cc'], extra_include_paths=paddle_includes, extra_cxx_cflags=extra_cc_args, extra_cuda_cflags=extra_nvcc_args, verbose=True)

class TestCustomSimpleSliceJit(unittest.TestCase):

    def test_slice_output(self):
        if False:
            while True:
                i = 10
        np_x = np.random.random((5, 2)).astype('float32')
        x = paddle.to_tensor(np_x)
        custom_op_out = custom_ops.custom_simple_slice(x, 2, 3)
        np_out = np_x[2:3]
        check_output(custom_op_out, np_out, 'out')
if __name__ == '__main__':
    unittest.main()