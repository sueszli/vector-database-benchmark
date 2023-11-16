import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base, incubate
from paddle.base import core
from paddle.pir_utils import test_with_pir_api
paddle.enable_static()

def _get_softmax_upper(x, fp16=True):
    if False:
        print('Hello World!')
    x_lower = np.tril(x)
    masked_x = np.where(x_lower == 0, -10000.0, x_lower).astype('float32')
    max_value = np.max(masked_x, axis=-1, keepdims=True)
    before_exp = masked_x - max_value
    exp = np.exp(before_exp)
    exp_sum = np.sum(exp, axis=-1, keepdims=True)
    rst = exp / exp_sum
    if fp16:
        rst = rst.astype('float16')
    return rst

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestSoftmaxMaskFuseOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'fused_softmax_mask_upper_triangle'
        self.python_api = paddle.incubate.softmax_mask_fuse_upper_triangle
        x = np.random.random((1, 4, 32, 32)).astype('float16')
        self.inputs = {'X': x}
        rst = _get_softmax_upper(x)
        self.outputs = {'Out': rst}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(core.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad_with_place(core.CUDAPlace(0), ['X'], 'Out', check_pir=True)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestSoftmaxMaskFuseOp1(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'fused_softmax_mask_upper_triangle'
        self.python_api = paddle.incubate.softmax_mask_fuse_upper_triangle
        x = np.random.random((1, 4, 32, 32))
        self.inputs = {'X': x}
        rst = _get_softmax_upper(x)
        self.outputs = {'Out': rst}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        try:
            self.check_output_with_place(core.CPUPlace(), check_pir=True)
        except (NotImplementedError, RuntimeError):
            pass

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        try:
            self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', check_pir=True)
        except (NotImplementedError, RuntimeError):
            pass

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestDropoutBiasFuseOp2(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(123)
        self.dtypes = ['float32', 'float16']

    @test_with_pir_api
    def test_static(self):
        if False:
            while True:
                i = 10
        for dtype in self.dtypes:
            with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
                input_x = paddle.static.data(name='x', shape=[1, 4, 32, 32], dtype=dtype)
                rst = incubate.softmax_mask_fuse_upper_triangle(input_x)
                x_in_np = np.random.random((1, 4, 32, 32)).astype(dtype)
                rst_np = _get_softmax_upper(x_in_np, dtype == 'float16')
                exe = base.Executor(base.CUDAPlace(0))
                fetches = exe.run(paddle.static.default_main_program(), feed={'x': x_in_np}, fetch_list=[rst])
                np.testing.assert_allclose(fetches[0], rst_np, rtol=1e-05)

    def test_dygraph(self):
        if False:
            return 10
        for dtype in self.dtypes:
            with base.dygraph.guard(base.CUDAPlace(0)):
                x_in_np = np.random.random((1, 4, 32, 32)).astype(dtype)
                rst_np = _get_softmax_upper(x_in_np, dtype == 'float16')
                input_x = base.dygraph.to_variable(x_in_np)
                rst = incubate.softmax_mask_fuse_upper_triangle(input_x)
                np.testing.assert_allclose(rst, rst_np, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()