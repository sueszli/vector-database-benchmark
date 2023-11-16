import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle import base
from paddle.base import Program, core, program_guard

class TestAddMMOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'addmm'
        self.python_api = paddle.addmm
        self.init_dtype_type()
        self.inputs = {'Input': np.random.random((100, 1)).astype(self.dtype), 'X': np.random.random((100, 10)).astype(self.dtype), 'Y': np.random.random((10, 20)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['Input'] + np.dot(self.inputs['X'], self.inputs['Y'])}

    def init_dtype_type(self):
        if False:
            return 10
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        self.check_grad(['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        if False:
            return 10
        self.check_grad(['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Input'], 'Out', no_grad_set=None)

class TestAddMMFP16Op(TestAddMMOp):

    def init_dtype_type(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16

    def test_check_output(self):
        if False:
            return 10
        self.check_output(atol=0.01)

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA or not support bfloat16')
class TestAddMMBF16Op(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'addmm'
        self.python_api = paddle.addmm
        self.init_dtype_type()
        self.inputs = {'Input': np.random.random((100, 1)).astype(self.np_dtype), 'X': np.random.random((100, 10)).astype(self.np_dtype), 'Y': np.random.random((10, 20)).astype(self.np_dtype)}
        self.outputs = {'Out': self.inputs['Input'] + np.dot(self.inputs['X'], self.inputs['Y'])}
        self.inputs['Input'] = convert_float_to_uint16(self.inputs['Input'])
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def init_dtype_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def test_check_output(self):
        if False:
            return 10
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        self.check_grad_with_place(self.place, ['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        if False:
            print('Hello World!')
        self.check_grad_with_place(self.place, ['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(self.place, ['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        if False:
            i = 10
            return i + 15
        self.check_grad_with_place(self.place, ['Input'], 'Out', no_grad_set=None)

class TestAddMMOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with program_guard(Program(), Program()):
            input = base.create_lod_tensor(np.array([[-1, -1], [-1, -1]]), [[2]], base.CPUPlace())
            x1 = base.create_lod_tensor(np.array([[-1, -1], [-1, -1]]), [[2]], base.CPUPlace())
            x2 = base.create_lod_tensor(np.array([[-1, -1], [-1, -1]]), [[2]], base.CPUPlace())
            self.assertRaises(TypeError, paddle.addmm, input, x1, x2)
            input = paddle.static.data(name='input', shape=[4, 4], dtype='int32')
            x3 = paddle.static.data(name='x3', shape=[4, 4], dtype='int32')
            x4 = paddle.static.data(name='x4', shape=[4, 4], dtype='int32')
            self.assertRaises(TypeError, paddle.addmm, input, x3, x4)
            x5 = paddle.static.data(name='x5', shape=[4, 5], dtype='float32')
            x6 = paddle.static.data(name='x6', shape=[4, 4], dtype='float32')
            self.assertRaises(ValueError, paddle.addmm, input, x5, x6)
            x7 = paddle.static.data(name='x7', shape=[4, 4], dtype='float32')
            x8 = paddle.static.data(name='x8', shape=[4, 4], dtype='float32')
            input1 = paddle.static.data(name='input1', shape=[2, 4], dtype='float32')
            self.assertRaises(ValueError, paddle.addmm, input1, x7, x8)
            x9 = paddle.static.data(name='x9', shape=[4, 4], dtype='float32')
            x10 = paddle.static.data(name='x10', shape=[4, 4], dtype='float32')
            input2 = paddle.static.data(name='input2', shape=[1, 2], dtype='float32')
            self.assertRaises(ValueError, paddle.addmm, input2, x9, x10)
            x11 = paddle.static.data(name='x11', shape=[4, 4], dtype='float32')
            x12 = paddle.static.data(name='x12', shape=[4, 4], dtype='float32')
            input3 = paddle.static.data(name='input3', shape=[4, 2], dtype='float32')
            self.assertRaises(ValueError, paddle.addmm, input3, x11, x12)
            x13 = paddle.static.data(name='x13', shape=[4, 4], dtype='float32')
            x14 = paddle.static.data(name='x14', shape=[4, 4], dtype='float32')
            input4 = paddle.static.data(name='input4', shape=[3, 1], dtype='float32')
            self.assertRaises(ValueError, paddle.addmm, input4, x13, x14)

class TestAddMMOp2(TestAddMMOp):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'addmm'
        self.python_api = paddle.addmm
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {'Input': np.random.random((20, 30)).astype(self.dtype), 'X': np.random.random((20, 6)).astype(self.dtype), 'Y': np.random.random((6, 30)).astype(self.dtype)}
        self.attrs = {'Alpha': 0.1, 'Beta': 1.0}
        self.outputs = {'Out': self.attrs['Beta'] * self.inputs['Input'] + self.attrs['Alpha'] * np.dot(self.inputs['X'], self.inputs['Y'])}

class TestAddMMOp3(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'addmm'
        self.python_api = paddle.addmm
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {'Input': np.random.random((1, 100)).astype(self.dtype), 'X': np.random.random((20, 10)).astype(self.dtype), 'Y': np.random.random((10, 100)).astype(self.dtype)}
        self.attrs = {'Alpha': 0.5, 'Beta': 2.0}
        self.outputs = {'Out': self.attrs['Beta'] * self.inputs['Input'] + self.attrs['Alpha'] * np.dot(self.inputs['X'], self.inputs['Y'])}

    def init_dtype_type(self):
        if False:
            return 10
        pass

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        if False:
            return 10
        self.check_grad(['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        if False:
            print('Hello World!')
        self.check_grad(['Input'], 'Out', no_grad_set=None)

class TestAddMMOp4(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'addmm'
        self.python_api = paddle.addmm
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {'Input': np.random.random(100).astype(self.dtype), 'X': np.random.random((20, 10)).astype(self.dtype), 'Y': np.random.random((10, 100)).astype(self.dtype)}
        self.attrs = {'Alpha': 0.5, 'Beta': 2.0}
        self.outputs = {'Out': self.attrs['Beta'] * self.inputs['Input'] + self.attrs['Alpha'] * np.dot(self.inputs['X'], self.inputs['Y'])}

    def init_dtype_type(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        if False:
            while True:
                i = 10
        self.check_grad(['Input'], 'Out', no_grad_set=None)

class TestAddMMOp5(unittest.TestCase):

    def test_api_with_dygraph(self):
        if False:
            print('Hello World!')
        np_input = np.random.random((20, 30)).astype(np.float32)
        np_x = np.random.random((20, 6)).astype(np.float32)
        np_y = np.random.random((6, 30)).astype(np.float32)
        with base.dygraph.guard():
            input = base.dygraph.to_variable(np_input)
            x = base.dygraph.to_variable(np_x)
            y = base.dygraph.to_variable(np_y)
            out = paddle.tensor.addmm(input, x, y)
            np.testing.assert_allclose(np_input + np.dot(np_x, np_y), out.numpy(), rtol=1e-05, atol=1e-08)

class TestAddMMAPI(unittest.TestCase):

    def test_api_error(self):
        if False:
            i = 10
            return i + 15
        data_x = np.ones((2, 2)).astype(np.float32)
        data_y = np.ones((2, 2)).astype(np.float32)
        data_input = np.ones((2, 2)).astype(np.float32)
        paddle.disable_static()

        def test_error1():
            if False:
                i = 10
                return i + 15
            data_x_wrong = np.ones((2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.addmm(input=input, x=x, y=y, beta=0.5, alpha=5.0)
        self.assertRaises(ValueError, test_error1)

        def test_error2():
            if False:
                print('Hello World!')
            data_x_wrong = np.ones(2).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.addmm(input=input, x=x, y=y, beta=0.5, alpha=5.0)
        self.assertRaises(ValueError, test_error2)

        def test_error3():
            if False:
                print('Hello World!')
            data_input_wrong = np.ones((2, 2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.addmm(input=input, x=x, y=y, beta=0.5, alpha=5.0)
        self.assertRaises(ValueError, test_error3)

        def test_error4():
            if False:
                i = 10
                return i + 15
            data_input_wrong = np.ones(5).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.addmm(input=input, x=x, y=y, beta=0.5, alpha=5.0)
        self.assertRaises(ValueError, test_error4)
        paddle.enable_static()

    def test_api_normal_1(self):
        if False:
            for i in range(10):
                print('nop')
        data_x = np.ones((2, 2)).astype(np.float32)
        data_y = np.ones((2, 2)).astype(np.float32)
        data_input = np.ones((2, 2)).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0
        paddle.disable_static()
        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)
        paddle_output = paddle.tensor.addmm(input=input, x=x, y=y, beta=data_beta, alpha=data_alpha)
        numpy_output = data_beta * data_input + data_alpha * np.dot(data_x, data_y)
        np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_api_normal_2(self):
        if False:
            for i in range(10):
                print('nop')
        data_x = np.ones((3, 10)).astype(np.float32)
        data_y = np.ones((10, 3)).astype(np.float32)
        data_input = np.ones(3).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0
        paddle.disable_static()
        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)
        paddle_output = paddle.tensor.addmm(input=input, x=x, y=y, beta=data_beta, alpha=data_alpha)
        numpy_output = data_beta * data_input + data_alpha * np.dot(data_x, data_y)
        np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_api_normal_3(self):
        if False:
            for i in range(10):
                print('nop')
        data_x = np.ones((3, 10)).astype(np.float32)
        data_y = np.ones((10, 3)).astype(np.float32)
        data_input = np.ones(1).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0
        paddle.disable_static()
        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)
        paddle_output = paddle.tensor.addmm(input=input, x=x, y=y, beta=data_beta, alpha=data_alpha)
        numpy_output = data_beta * data_input + data_alpha * np.dot(data_x, data_y)
        np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=1e-05)
        paddle.enable_static()
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()