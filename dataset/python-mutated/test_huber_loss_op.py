import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle.base import core

def huber_loss_forward(val, delta):
    if False:
        for i in range(10):
            print('nop')
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)

def huber_loss_wraper(x, y, delta):
    if False:
        return 10
    a = paddle._C_ops.huber_loss(x, y, delta)
    return a

class TestHuberLossOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'huber_loss'
        self.python_out_sig = ['Out']
        self.python_api = huber_loss_wraper
        self.delta = 1.0
        self.init_dtype()
        self.init_input()
        shape = self.set_shape()
        residual = self.inputs['Y'] - self.inputs['X']
        loss = np.vectorize(huber_loss_forward)(residual, self.delta).astype(self.dtype)
        self.attrs = {'delta': self.delta}
        self.outputs = {'Residual': residual, 'Out': loss.reshape(shape)}

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float32

    def init_input(self):
        if False:
            while True:
                i = 10
        shape = self.set_shape()
        self.inputs = {'X': np.random.uniform(0, 1.0, shape).astype(self.dtype), 'Y': np.random.uniform(0, 1.0, shape).astype(self.dtype)}

    def set_shape(self):
        if False:
            i = 10
            return i + 15
        return (100, 1)

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        if False:
            return 10
        self.check_grad(['Y'], 'Out', no_grad_set=set('residual'))

    def test_check_grad_ingore_y(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out', no_grad_set=set('residual'))

def TestHuberLossOp1(TestHuberLossOp):
    if False:
        i = 10
        return i + 15

    def set_shape(self):
        if False:
            print('Hello World!')
        return 64

def TestHuberLossOp2(TestHuberLossOp):
    if False:
        for i in range(10):
            print('nop')

    def set_shape(self):
        if False:
            for i in range(10):
                print('nop')
        return (6, 6)

def TestHuberLossOp3(TestHuberLossOp):
    if False:
        for i in range(10):
            print('nop')

    def set_shape(self):
        if False:
            while True:
                i = 10
        return (6, 6, 1)

class TestHuberLossFP16Op(TestHuberLossOp):

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA or not support bfloat16')
class TestHuberLossBF16Op(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'huber_loss'
        self.python_out_sig = ['Out']
        self.python_api = huber_loss_wraper
        self.delta = 1.0
        self.init_dtype()
        self.init_input()
        shape = self.set_shape()
        residual = self.inputs['Y'] - self.inputs['X']
        loss = np.vectorize(huber_loss_forward)(residual, self.delta).astype(self.np_dtype)
        self.attrs = {'delta': self.delta}
        self.outputs = {'Residual': residual, 'Out': loss.reshape(shape)}
        self.place = core.CUDAPlace(0)
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Residual'] = convert_float_to_uint16(self.outputs['Residual'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])

    def init_dtype(self):
        if False:
            return 10
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def init_input(self):
        if False:
            print('Hello World!')
        shape = self.set_shape()
        self.inputs = {'X': np.random.uniform(0, 1.0, shape).astype(self.np_dtype), 'Y': np.random.uniform(0, 1.0, shape).astype(self.np_dtype)}

    def set_shape(self):
        if False:
            for i in range(10):
                print('nop')
        return (100, 1)

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(self.place, ['Y'], 'Out', no_grad_set=set('residual'))

    def test_check_grad_ingore_y(self):
        if False:
            print('Hello World!')
        self.check_grad_with_place(self.place, ['X'], 'Out', no_grad_set=set('residual'))
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()