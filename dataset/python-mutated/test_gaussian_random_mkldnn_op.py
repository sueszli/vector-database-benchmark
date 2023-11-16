import unittest
import numpy as np
from op_test import OpTest
from test_gaussian_random_op import TestGaussianRandomOp
import paddle

class TestMKLDNNGaussianRandomOpSeed10(TestGaussianRandomOp):

    def init_kernel_type(self):
        if False:
            i = 10
            return i + 15
        self.use_mkldnn = True

class TestMKLDNNGaussianRandomOpSeed0(TestGaussianRandomOp):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        TestGaussianRandomOp.setUp(self)
        self.use_mkldnn = True
        self.attrs = {'shape': [123, 92], 'mean': 1.0, 'std': 2.0, 'seed': 10, 'use_mkldnn': self.use_mkldnn}

class TestGaussianRandomOp_ZeroDim(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'gaussian_random'
        self.__class__.op_type = 'gaussian_random'
        self.python_api = paddle.normal
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = True
        self.attrs = {'shape': [], 'mean': self.mean, 'std': self.std, 'seed': 10, 'use_mkldnn': self.use_mkldnn}
        paddle.seed(10)
        self.outputs = {'Out': np.random.normal(self.mean, self.std, ())}

    def set_attrs(self):
        if False:
            while True:
                i = 10
        self.mean = 1.0
        self.std = 2.0

    def test_check_output(self):
        if False:
            while True:
                i = 10
        pass

    def test_check_grad(self):
        if False:
            return 10
        pass
if __name__ == '__main__':
    unittest.main()