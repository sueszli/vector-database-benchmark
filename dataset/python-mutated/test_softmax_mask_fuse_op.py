import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base, incubate
from paddle.base import core
paddle.enable_static()

def _get_softmax(x, mask, fp16=True):
    if False:
        i = 10
        return i + 15
    masked_x = (x + mask).astype('float32')
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
            print('Hello World!')
        self.op_type = 'fused_softmax_mask'
        self.python_api = paddle.incubate.softmax_mask_fuse
        x = np.random.random((1, 1, 8, 32))
        mask = np.random.randint(0, 2, (1, 1, 8, 32))
        mask_input = np.where(mask == 1, -10000.0, mask)
        self.inputs = {'X': x, 'Mask': mask_input}
        rst = _get_softmax(x, mask_input)
        self.outputs = {'Out': rst}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out')

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestSoftmaxMaskFuseOp0(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'fused_softmax_mask'
        self.python_api = paddle.incubate.softmax_mask_fuse
        x = np.random.random((1, 1, 8, 32)).astype('float16')
        mask = np.random.randint(0, 2, (1, 1, 8, 32)).astype('float16')
        mask_input = np.where(mask == 1, -10000.0, mask)
        self.inputs = {'X': x, 'Mask': mask_input}
        rst = _get_softmax(x, mask_input)
        self.outputs = {'Out': rst}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad_with_place(core.CUDAPlace(0), ['X'], 'Out')

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestSoftmaxMaskFuseOp01(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'fused_softmax_mask'
        self.python_api = paddle.incubate.softmax_mask_fuse
        x = np.random.random((1, 1, 8, 32)).astype('float16')
        mask = np.random.randint(0, 2, (1, 1, 8, 32)).astype('float32')
        mask_input = np.where(mask == 1, -10000.0, mask)
        self.inputs = {'X': x, 'Mask': mask_input}
        rst = _get_softmax(x, mask_input)
        self.outputs = {'Out': rst}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad_with_place(core.CUDAPlace(0), ['X'], 'Out')

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestDropoutBiasFuseOp3(unittest.TestCase):

    def test_static_result(self):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(base.Program(), base.Program()):
            input_x = paddle.static.data(name='x', shape=[1, 1, 8, 32], dtype='float32')
            input_mask = paddle.static.data(name='mask', shape=[1, 1, 8, 32], dtype='float32')
            rst = incubate.softmax_mask_fuse(input_x, input_mask)
            x_in_np = np.random.random((1, 1, 8, 32)).astype('float32')
            mask = np.random.randint(0, 2, (1, 1, 8, 32)).astype('float32')
            mask_in_np = np.where(mask == 1, -10000.0, mask)
            rst_np = _get_softmax(x_in_np, mask_in_np, False)
            exe = base.Executor(base.CUDAPlace(0))
            fetches = exe.run(base.default_main_program(), feed={'x': x_in_np, 'mask': mask_in_np}, fetch_list=[rst])
            np.testing.assert_allclose(fetches[0], rst_np, rtol=1e-05)

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard(base.CUDAPlace(0)):
            x_in_np = np.random.random((1, 1, 8, 32)).astype('float32')
            mask = np.random.randint(0, 2, (1, 1, 8, 32)).astype('float32')
            mask_in_np = np.where(mask == 1, -10000.0, mask)
            rst_np = _get_softmax(x_in_np, mask_in_np, False)
            input_x = base.dygraph.to_variable(x_in_np)
            input_mask = base.dygraph.to_variable(mask_in_np)
            rst = incubate.softmax_mask_fuse(input_x, input_mask)
            np.testing.assert_allclose(rst, rst_np, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()