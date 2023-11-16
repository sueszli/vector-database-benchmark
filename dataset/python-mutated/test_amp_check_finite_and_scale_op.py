import unittest
import numpy as np
from op_test import OpTest
from paddle.static.amp import amp_nn

def check_finite_and_unscale_wrapper(x, scale):
    if False:
        return 10
    (_, found_inf) = amp_nn.check_finite_and_unscale([x], scale)
    return (x, found_inf)

class TestCheckFiniteAndUnscaleOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'check_finite_and_unscale'
        self.python_api = check_finite_and_unscale_wrapper
        self.python_out_sig = ['out0', 'FoundInfinite']
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        scale = np.random.random(1).astype(self.dtype)
        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {'FoundInfinite': np.array([0]), 'Out': [('out0', x / scale)]}

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float32

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output()

class TestCheckFiniteAndUnscaleOpWithNan(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'check_finite_and_unscale'
        self.init_dtype()
        self.python_api = check_finite_and_unscale_wrapper
        self.python_out_sig = ['out0', 'FoundInfinite']
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.nan
        scale = np.random.random(1).astype(self.dtype)
        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {'FoundInfinite': np.array([1]), 'Out': [('out0', x)]}

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float32

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(no_check_set=['Out'])

class TestCheckFiniteAndUnscaleOpWithInf(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'check_finite_and_unscale'
        self.init_dtype()
        self.python_api = check_finite_and_unscale_wrapper
        self.python_out_sig = ['out0', 'FoundInfinite']
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.inf
        scale = np.random.random(1).astype(self.dtype)
        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {'FoundInfinite': np.array([1]), 'Out': [('out0', x)]}

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float32

    def test_check_output(self):
        if False:
            return 10
        self.check_output(no_check_set=['Out'])
if __name__ == '__main__':
    unittest.main()