import unittest
import gradient_checker
import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle import base
from paddle.base import Program, core, program_guard
from paddle.pir_utils import test_with_pir_api

class TestSignOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'sign'
        self.python_api = paddle.sign
        self.inputs = {'X': np.random.uniform(-10, 10, (10, 10)).astype('float64')}
        self.outputs = {'Out': np.sign(self.inputs['X'])}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_pir=True)

class TestSignFP16Op(TestSignOp):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'sign'
        self.python_api = paddle.sign
        self.inputs = {'X': np.random.uniform(-10, 10, (10, 10)).astype('float16')}
        self.outputs = {'Out': np.sign(self.inputs['X'])}

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA or not support bfloat16')
class TestSignBF16Op(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'sign'
        self.python_api = paddle.sign
        self.dtype = np.uint16
        self.inputs = {'X': np.random.uniform(-10, 10, (10, 10)).astype('float32')}
        self.outputs = {'Out': np.sign(self.inputs['X'])}
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(self.place, check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad_with_place(self.place, ['X'], 'Out', check_pir=True)

class TestSignAPI(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.place = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(base.CUDAPlace(0))

    def test_dygraph(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():
            np_x = np.array([-1.0, 0.0, -0.0, 1.2, 1.5], dtype='float64')
            x = paddle.to_tensor(np_x)
            z = paddle.sign(x)
            np_z = z.numpy()
            z_expected = np.sign(np_x)
            self.assertEqual((np_z == z_expected).all(), True)

    def test_static(self):
        if False:
            i = 10
            return i + 15
        np_input2 = np.random.uniform(-10, 10, (12, 10)).astype('int16')
        np_input3 = np.random.uniform(-10, 10, (12, 10)).astype('int32')
        np_input4 = np.random.uniform(-10, 10, (12, 10)).astype('int64')
        np_out2 = np.sign(np_input2)
        np_out3 = np.sign(np_input3)
        np_out4 = np.sign(np_input4)

        def run(place):
            if False:
                print('Hello World!')
            with program_guard(Program(), Program()):
                input1 = 12
                self.assertRaises(TypeError, paddle.tensor.math.sign, input1)
                input2 = paddle.static.data(name='input2', shape=[12, 10], dtype='int16')
                input3 = paddle.static.data(name='input3', shape=[12, 10], dtype='int32')
                input4 = paddle.static.data(name='input4', shape=[12, 10], dtype='int64')
                out2 = paddle.sign(input2)
                out3 = paddle.sign(input3)
                out4 = paddle.sign(input4)
                exe = paddle.static.Executor(place)
                (res2, res3, res4) = exe.run(paddle.static.default_main_program(), feed={'input2': np_input2, 'input3': np_input3, 'input4': np_input4}, fetch_list=[out2, out3, out4])
                self.assertEqual((res2 == np_out2).all(), True)
                self.assertEqual((res3 == np_out3).all(), True)
                self.assertEqual((res4 == np_out4).all(), True)
                input5 = paddle.static.data(name='input5', shape=[-1, 4], dtype='float16')
                paddle.sign(input5)
        for place in self.place:
            run(place)

class TestSignDoubleGradCheck(unittest.TestCase):

    def sign_wrapper(self, x):
        if False:
            while True:
                i = 10
        return paddle.sign(x[0])

    @test_with_pir_api
    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        eps = 0.005
        dtype = np.float32
        data = paddle.static.data('data', [1, 4], dtype)
        data.persistable = True
        out = paddle.sign(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)
        gradient_checker.double_grad_check([data], out, x_init=[data_arr], place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.sign_wrapper, [data], out, x_init=[data_arr], place=place)

    def test_grad(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestSignTripleGradCheck(unittest.TestCase):

    def sign_wrapper(self, x):
        if False:
            print('Hello World!')
        return paddle.sign(x[0])

    @test_with_pir_api
    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        eps = 0.005
        dtype = np.float32
        data = paddle.static.data('data', [1, 4], dtype)
        data.persistable = True
        out = paddle.sign(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)
        gradient_checker.triple_grad_check([data], out, x_init=[data_arr], place=place, eps=eps)
        gradient_checker.triple_grad_check_for_dygraph(self.sign_wrapper, [data], out, x_init=[data_arr], place=place)

    def test_grad(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()