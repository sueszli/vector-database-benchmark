import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

class TestUnStackOpBase(OpTest):

    def initDefaultParameters(self):
        if False:
            i = 10
            return i + 15
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'float64'

    def initParameters(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_y_names(self):
        if False:
            print('Hello World!')
        y_names = []
        for i in range(self.input_dim[self.axis]):
            y_names.append(f'y{i}')
        return y_names

    def setUp(self):
        if False:
            print('Hello World!')
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'unstack'
        self.python_api = paddle.unstack
        self.x = np.random.random(size=self.input_dim).astype(self.dtype)
        outs = np.split(self.x, self.input_dim[self.axis], self.axis)
        new_shape = list(self.input_dim)
        del new_shape[self.axis]
        y_names = self.get_y_names()
        tmp = []
        tmp_names = []
        for i in range(self.input_dim[self.axis]):
            tmp.append((y_names[i], np.reshape(outs[i], new_shape)))
            tmp_names.append(y_names[i])
        self.python_out_sig = tmp_names
        self.inputs = {'X': self.x}
        self.outputs = {'Y': tmp}
        self.attrs = {'axis': self.axis, 'num': self.input_dim[self.axis]}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], self.get_y_names(), check_pir=True)

class TestUnStackFP16Op(TestUnStackOpBase):

    def initParameters(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16

class TestStackFP16Op3(TestUnStackOpBase):

    def initParameters(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float16
        self.axis = -1

class TestStackFP16Op4(TestUnStackOpBase):

    def initParameters(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float16
        self.axis = -3

class TestStackFP16Op5(TestUnStackOpBase):

    def initParameters(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16
        self.axis = 1

class TestStackFP16Op6(TestUnStackOpBase):

    def initParameters(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16
        self.axis = 2

class TestStackOp3(TestUnStackOpBase):

    def initParameters(self):
        if False:
            print('Hello World!')
        self.axis = -1

class TestStackOp4(TestUnStackOpBase):

    def initParameters(self):
        if False:
            while True:
                i = 10
        self.axis = -3

class TestStackOp5(TestUnStackOpBase):

    def initParameters(self):
        if False:
            print('Hello World!')
        self.axis = 1

class TestStackOp6(TestUnStackOpBase):

    def initParameters(self):
        if False:
            return 10
        self.axis = 2

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA and do not support bfloat16')
class TestUnStackBF16Op(OpTest):

    def initDefaultParameters(self):
        if False:
            while True:
                i = 10
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = np.uint16

    def initParameters(self):
        if False:
            return 10
        pass

    def get_y_names(self):
        if False:
            print('Hello World!')
        y_names = []
        for i in range(self.input_dim[self.axis]):
            y_names.append(f'y{i}')
        return y_names

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'unstack'
        self.python_api = paddle.unstack
        self.x = np.random.random(size=self.input_dim).astype(np.float32)
        outs = np.split(self.x, self.input_dim[self.axis], self.axis)
        new_shape = list(self.input_dim)
        del new_shape[self.axis]
        y_names = self.get_y_names()
        tmp = []
        tmp_names = []
        for i in range(self.input_dim[self.axis]):
            tmp.append((y_names[i], np.reshape(convert_float_to_uint16(outs[i]), new_shape)))
            tmp_names.append(y_names[i])
        self.x = convert_float_to_uint16(self.x)
        self.python_out_sig = tmp_names
        self.inputs = {'X': self.x}
        self.outputs = {'Y': tmp}
        self.attrs = {'axis': self.axis, 'num': self.input_dim[self.axis]}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            x = paddle.to_tensor(self.inputs['X'])
            x.stop_gradient = False
            y = paddle.unstack(x, axis=self.attrs['axis'], num=self.attrs['num'])
            dx = paddle.grad(y, x)[0].numpy()
            dx_expected = convert_float_to_uint16(np.ones(self.input_dim, np.float32))
            np.testing.assert_array_equal(dx, dx_expected)

class TestUnstackZeroInputOp(unittest.TestCase):

    @test_with_pir_api
    def unstack_zero_input_static(self):
        if False:
            return 10
        paddle.enable_static()
        array = np.array([], dtype=np.float32)
        x = paddle.to_tensor(np.reshape(array, [0]), dtype='float32')
        paddle.unstack(x, axis=1)

    def unstack_zero_input_dynamic(self):
        if False:
            print('Hello World!')
        array = np.array([], dtype=np.float32)
        x = paddle.to_tensor(np.reshape(array, [0]), dtype='float32')
        paddle.unstack(x, axis=1)

    def test_type_error(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        self.assertRaises(ValueError, self.unstack_zero_input_dynamic)
        self.assertRaises(ValueError, self.unstack_zero_input_static)
        paddle.disable_static()
if __name__ == '__main__':
    unittest.main()