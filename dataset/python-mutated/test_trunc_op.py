import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api
paddle.enable_static()

class TestTruncOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'trunc'
        self.python_api = paddle.trunc
        self.init_dtype_type()
        np.random.seed(2021)
        self.inputs = {'X': np.random.random((20, 20)).astype(self.dtype)}
        self.outputs = {'Out': np.trunc(self.inputs['X'])}

    def init_dtype_type(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', numeric_grad_delta=1e-05, check_pir=True)

class TestFloatTruncOp(TestTruncOp):

    def init_dtype_type(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float32
        self.__class__.exist_fp64_check_grad = True

    def test_check_grad(self):
        if False:
            print('Hello World!')
        pass

class TestIntTruncOp(TestTruncOp):

    def init_dtype_type(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int32
        self.__class__.exist_fp64_check_grad = True

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        pass

class TestTruncAPI(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [20, 20]
        self.x = np.random.random((20, 20)).astype(np.float32)
        self.place = paddle.CPUPlace()

    @test_with_pir_api
    def test_api_static(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            out = paddle.trunc(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = np.trunc(self.x)
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=1e-08)

    def test_api_dygraph(self):
        if False:
            while True:
                i = 10
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.trunc(x_tensor)
        out_ref = np.trunc(self.x)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-08)
        paddle.enable_static()

    def test_errors(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [20, 20], 'bool')
            self.assertRaises(TypeError, paddle.trunc, x)

class TestTruncFP16OP(TestTruncOp):

    def init_dtype_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestTruncBF16OP(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.python_api = paddle.trunc
        self.op_type = 'trunc'
        self.dtype = np.uint16
        np.random.seed(2021)
        x = np.random.random((20, 20)).astype('float32')
        out = np.trunc(x)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', numeric_grad_delta=1e-05, check_pir=True)
if __name__ == '__main__':
    unittest.main()