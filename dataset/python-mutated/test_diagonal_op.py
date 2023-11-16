import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api
paddle.enable_static()

class TestDiagonalOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'diagonal'
        self.python_api = paddle.diagonal
        self.init_dtype()
        self.init_config()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['Input'], 'Out', check_pir=True)

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = 'float64'

    def init_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.case = np.random.randn(10, 5, 2).astype(self.dtype)
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.target = np.diagonal(self.inputs['Input'], offset=self.attrs['offset'], axis1=self.attrs['axis1'], axis2=self.attrs['axis2'])

class TestDiagonalOpCase1(TestDiagonalOp):

    def init_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.case = np.random.randn(4, 2, 4, 4).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -2, 'axis1': 3, 'axis2': 0}
        self.target = np.diagonal(self.inputs['Input'], offset=self.attrs['offset'], axis1=self.attrs['axis1'], axis2=self.attrs['axis2'])

class TestDiagonalOpCase2(TestDiagonalOp):

    def init_config(self):
        if False:
            i = 10
            return i + 15
        self.case = np.random.randn(100, 100).astype('int64')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.target = np.diagonal(self.inputs['Input'], offset=self.attrs['offset'], axis1=self.attrs['axis1'], axis2=self.attrs['axis2'])
        self.grad_x = np.eye(100).astype('int64')
        self.grad_out = np.ones(100).astype('int64')

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['Input'], 'Out', user_defined_grads=[self.grad_x], user_defined_grad_outputs=[self.grad_out], check_pir=True)

class TestDiagonalOpCase3(TestDiagonalOp):

    def init_config(self):
        if False:
            print('Hello World!')
        self.case = np.random.randint(0, 2, (4, 2, 4, 4)).astype('bool')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -2, 'axis1': 3, 'axis2': 0}
        self.target = np.diagonal(self.inputs['Input'], offset=self.attrs['offset'], axis1=self.attrs['axis1'], axis2=self.attrs['axis2'])

    def test_check_grad(self):
        if False:
            print('Hello World!')
        pass

class TestDiagonalOpCase4(TestDiagonalOp):

    def init_config(self):
        if False:
            i = 10
            return i + 15
        self.case = np.random.randn(100, 100).astype('int64')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 1, 'axis1': 1, 'axis2': 0}
        self.target = np.diagonal(self.inputs['Input'], offset=self.attrs['offset'], axis1=self.attrs['axis1'], axis2=self.attrs['axis2'])

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestDiagonalOpCase5(TestDiagonalOp):

    def init_config(self):
        if False:
            i = 10
            return i + 15
        self.case = np.random.randn(4, 2, 4, 4).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -2, 'axis1': 0, 'axis2': 3}
        self.target = np.diagonal(self.inputs['Input'], offset=self.attrs['offset'], axis1=self.attrs['axis1'], axis2=self.attrs['axis2'])

class TestDiagonalAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = [10, 3, 4]
        self.x = np.random.random((10, 3, 4)).astype(np.float32)
        self.place = paddle.CPUPlace()

    @test_with_pir_api
    def test_api_static(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            out = paddle.diagonal(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = np.diagonal(self.x)
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=1e-08)

    def test_api_dygraph(self):
        if False:
            print('Hello World!')
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.diagonal(x_tensor)
        out_ref = np.diagonal(self.x)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-08)
        paddle.enable_static()

    def test_api_eager(self):
        if False:
            print('Hello World!')
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.diagonal(x_tensor)
        out2 = paddle.diagonal(x_tensor, offset=0, axis1=2, axis2=1)
        out3 = paddle.diagonal(x_tensor, offset=1, axis1=0, axis2=1)
        out4 = paddle.diagonal(x_tensor, offset=0, axis1=1, axis2=2)
        out_ref = np.diagonal(self.x)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-08)
        out2_ref = np.diagonal(self.x, offset=0, axis1=2, axis2=1)
        np.testing.assert_allclose(out2.numpy(), out2_ref, rtol=1e-08)
        out3_ref = np.diagonal(self.x, offset=1, axis1=0, axis2=1)
        np.testing.assert_allclose(out3.numpy(), out3_ref, rtol=1e-08)
        out4_ref = np.diagonal(self.x, offset=0, axis1=1, axis2=2)
        np.testing.assert_allclose(out4.numpy(), out4_ref, rtol=1e-08)
        paddle.enable_static()

class TestDiagonalFP16OP(TestDiagonalOp):

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestDiagonalBF16OP(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'diagonal'
        self.python_api = paddle.diagonal
        self.dtype = np.uint16
        self.init_config()
        self.outputs = {'Out': convert_float_to_uint16(self.target)}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            return 10
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['Input'], 'Out', check_pir=True)

    def init_config(self):
        if False:
            while True:
                i = 10
        self.case = np.random.randn(10, 5, 2).astype(np.float32)
        self.inputs = {'Input': convert_float_to_uint16(self.case)}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.target = np.diagonal(self.case, offset=self.attrs['offset'], axis1=self.attrs['axis1'], axis2=self.attrs['axis2']).copy()
if __name__ == '__main__':
    unittest.main()