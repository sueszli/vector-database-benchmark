import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestScaleOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_shape()
        self.op_type = 'scale'
        self.inputs = {'X': np.random.random(self.shape).astype(np.float32)}
        self.attrs = {'scale': -2.3, 'use_mkldnn': True, 'bias': 0.2}
        self.use_mkldnn = True
        self.outputs = {'Out': self.inputs['X'] * self.attrs['scale'] + self.attrs['bias']}

    def init_shape(self):
        if False:
            return 10
        self.shape = [10, 10]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestScaleOp_ZeroDim(TestScaleOp):

    def init_shape(self):
        if False:
            return 10
        self.shape = []

class TestScaleOpBiasNotAfterScale(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'scale'
        self.inputs = {'X': np.random.random((10, 10)).astype(np.float32)}
        self.attrs = {'scale': 1.5, 'use_mkldnn': True, 'bias': 2.3, 'bias_after_scale': False}
        self.use_mkldnn = True
        self.outputs = {'Out': (self.inputs['X'] + self.attrs['bias']) * self.attrs['scale']}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestScaleOpScaleTensor(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'scale'
        self.scale = -2.3
        self.inputs = {'X': np.random.random((10, 10)).astype(np.float32), 'ScaleTensor': np.array([self.scale]).astype(np.float32)}
        self.attrs = {}
        self.outputs = {'Out': self.inputs['X'] * self.scale}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestScaleOpScaleTensorNotBiasAfterScale(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'scale'
        self.scale = -1.2
        self.inputs = {'X': np.random.random((10, 10)).astype(np.float32), 'ScaleTensor': np.array([self.scale]).astype(np.float32)}
        self.attrs = {'bias': -6.8, 'bias_after_scale': False}
        self.outputs = {'Out': (self.inputs['X'] + self.attrs['bias']) * self.inputs['ScaleTensor']}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', check_dygraph=False)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()