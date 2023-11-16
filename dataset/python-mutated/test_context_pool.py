import os
import unittest
import numpy as np
from utils import extra_cc_args, extra_nvcc_args, paddle_includes
import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
file = f'{get_build_directory()}\\context_pool_jit\\context_pool_jit.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)
custom_ops = load(name='context_pool_jit', sources=['context_pool_test_op.cc'], extra_include_paths=paddle_includes, extra_cxx_cflags=extra_cc_args, extra_cuda_cflags=extra_nvcc_args, verbose=True)

class TestContextPool(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.devices = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')

    def test_use_context_pool(self):
        if False:
            return 10
        x = paddle.ones([2, 2], dtype='float32')
        out = custom_ops.context_pool_test(x)
        np.testing.assert_array_equal(x.numpy(), out.numpy())
if __name__ == '__main__':
    unittest.main()