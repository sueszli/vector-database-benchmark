import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

def ref_selu(x, scale=1.0507009873554805, alpha=1.6732632423543772):
    if False:
        i = 10
        return i + 15
    out = np.copy(x)
    out_flat = out.flatten()
    for i in range(out_flat.size):
        if out_flat[i] < 0:
            out_flat[i] = alpha * np.exp(out_flat[i]) - alpha
        out_flat[i] = scale * out_flat[i]
    out = out_flat.reshape(x.shape)
    return out

class SeluTest(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'selu'
        self.python_api = paddle.nn.functional.selu
        self.x_shape = [3, 5, 5, 10]
        self.init_x_shape()
        self.init_dtype()
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        if self.dtype == np.uint16:
            x = np.random.normal(size=self.x_shape).astype(np.float32)
        else:
            x = np.random.normal(size=self.x_shape).astype(self.dtype)
        x[np.abs(x) < 0.005] = 0.02
        out = ref_selu(x, scale, alpha)
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(x)}
            self.outputs = {'Out': convert_float_to_uint16(out)}
        else:
            self.inputs = {'X': x}
            self.outputs = {'Out': out}
        self.attrs = {'alpha': alpha, 'scale': scale}

    def init_x_shape(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out', check_pir=True)

class SeluTestFP16OP(SeluTest):

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA and do not support bfloat16')
class SeluTestBF16OP(SeluTest):

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.uint16

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(core.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad_with_place(core.CUDAPlace(0), ['X'], 'Out', check_pir=True)

class TestSeluAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.scale = 1.5
        self.alpha = 2.0
        self.x_np = np.random.normal(size=[3, 5, 5, 10]).astype(np.float64)
        self.x_np[np.abs(self.x_np) < 0.005] = 0.02
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

    @test_with_pir_api
    def test_static_api(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.selu(x, self.scale, self.alpha)
            selu = paddle.nn.SELU(self.scale, self.alpha)
            out2 = selu(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_selu(self.x_np, self.scale, self.alpha)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        if False:
            print('Hello World!')
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.selu(x, self.scale, self.alpha)
        selu = paddle.nn.SELU(self.scale, self.alpha)
        out2 = selu(x)
        out_ref = ref_selu(self.x_np, self.scale, self.alpha)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    @test_with_pir_api
    def test_base_api(self):
        if False:
            return 10
        with base.program_guard(base.Program()):
            x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            out = F.selu(x, self.scale, self.alpha)
            exe = base.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_selu(self.x_np, self.scale, self.alpha)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_errors(self):
        if False:
            return 10
        with paddle.static.program_guard(paddle.static.Program()):
            self.assertRaises(TypeError, F.selu, 1)
            x_int32 = paddle.static.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.selu, x_int32)
            x_fp32 = paddle.static.data(name='x_fp32', shape=[12, 10], dtype='float32')
            self.assertRaises(ValueError, F.selu, x_fp32, -1.0)
            self.assertRaises(ValueError, F.selu, x_fp32, 1.6, -1.0)
            x_fp16 = paddle.static.data(name='x_fp16', shape=[12, 10], dtype='float16')
            F.selu(x_fp16)
if __name__ == '__main__':
    unittest.main()