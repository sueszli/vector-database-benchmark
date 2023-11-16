"""
Unit testing for affine_channel_op
"""
import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle
from paddle.base import core

def affine_channel(x, scale, bias, layout):
    if False:
        for i in range(10):
            print('nop')
    C = x.shape[1] if layout == 'NCHW' else x.shape[-1]
    if len(x.shape) == 4:
        new_shape = (1, C, 1, 1) if layout == 'NCHW' else (1, 1, 1, C)
    else:
        new_shape = (1, C)
    scale = scale.reshape(new_shape)
    bias = bias.reshape(new_shape)
    return x * scale + bias

class TestAffineChannelOp(XPUOpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'affine_channel'
        self.init_test_case()
        x = np.random.random(self.shape).astype('float32')
        scale = np.random.random(self.C).astype('float32')
        bias = np.random.random(self.C).astype('float32')
        y = affine_channel(x, scale, bias, self.layout)
        self.inputs = {'X': x, 'Scale': scale, 'Bias': bias}
        self.attrs = {'data_layout': self.layout}
        self.outputs = {'Out': y}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X', 'Scale', 'Bias'], 'Out')

    def test_check_grad_stopgrad_dx(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['Scale', 'Bias'], 'Out', no_grad_set=set('X'))

    def test_check_grad_stopgrad_dscale_dbias(self):
        if False:
            i = 10
            return i + 15
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out', no_grad_set={'Scale', 'Bias'})

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.shape = [2, 100, 3, 3]
        self.C = 100
        self.layout = 'NCHW'

class TestAffineChannelNHWC(TestAffineChannelOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.shape = [2, 3, 3, 100]
        self.C = 100
        self.layout = 'NHWC'

    def test_check_grad_stopgrad_dx(self):
        if False:
            i = 10
            return i + 15
        return

    def test_check_grad_stopgrad_dscale_dbias(self):
        if False:
            for i in range(10):
                print('nop')
        return

class TestAffineChannel2D(TestAffineChannelOp):

    def init_test_case(self):
        if False:
            return 10
        self.shape = [2, 100]
        self.C = 100
        self.layout = 'NCHW'

    def test_check_grad_stopgrad_dx(self):
        if False:
            for i in range(10):
                print('nop')
        return

    def test_check_grad_stopgrad_dscale_dbias(self):
        if False:
            i = 10
            return i + 15
        return
if __name__ == '__main__':
    unittest.main()