"""
Unit testing for affine_channel_op
"""
import unittest
import numpy as np
from op_test import OpTest

def affine_channel(x, scale, bias, layout):
    if False:
        i = 10
        return i + 15
    C = x.shape[1] if layout == 'NCHW' else x.shape[-1]
    if len(x.shape) == 4:
        new_shape = (1, C, 1, 1) if layout == 'NCHW' else (1, 1, 1, C)
    else:
        new_shape = (1, C)
    scale = scale.reshape(new_shape)
    bias = bias.reshape(new_shape)
    return x * scale + bias

class TestAffineChannelOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'affine_channel'
        self.init_test_case()
        x = np.random.random(self.shape).astype('float64')
        scale = np.random.random(self.C).astype('float64')
        bias = np.random.random(self.C).astype('float64')
        y = affine_channel(x, scale, bias, self.layout)
        self.inputs = {'X': x, 'Scale': scale, 'Bias': bias}
        self.attrs = {'data_layout': self.layout}
        self.outputs = {'Out': y}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X', 'Scale', 'Bias'], 'Out', check_dygraph=False)

    def test_check_grad_stopgrad_dx(self):
        if False:
            print('Hello World!')
        self.check_grad(['Scale', 'Bias'], 'Out', no_grad_set=set('X'), check_dygraph=False)

    def test_check_grad_stopgrad_dscale_dbias(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', no_grad_set={'Scale', 'Bias'}, check_dygraph=False)

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [2, 100, 3, 3]
        self.C = 100
        self.layout = 'NCHW'

class TestAffineChannelNHWC(TestAffineChannelOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [2, 3, 3, 100]
        self.C = 100
        self.layout = 'NHWC'

    def test_check_grad_stopgrad_dx(self):
        if False:
            print('Hello World!')
        return

    def test_check_grad_stopgrad_dscale_dbias(self):
        if False:
            while True:
                i = 10
        return

class TestAffineChannel2D(TestAffineChannelOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.shape = [2, 100]
        self.C = 100
        self.layout = 'NCHW'

    def test_check_grad_stopgrad_dx(self):
        if False:
            print('Hello World!')
        return

    def test_check_grad_stopgrad_dscale_dbias(self):
        if False:
            while True:
                i = 10
        return
if __name__ == '__main__':
    unittest.main()