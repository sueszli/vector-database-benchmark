import unittest
from unittest import TestCase
import numpy as np
import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F

class TestFunctionalConv1DError(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.input = []
        self.filter = []
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.data_format = 'NCL'

    def dygraph_case(self):
        if False:
            return 10
        with dg.guard():
            x = dg.to_variable(self.input, dtype=paddle.float32)
            w = dg.to_variable(self.filter, dtype=paddle.float32)
            b = None if self.bias is None else dg.to_variable(self.bias, dtype=paddle.float32)
            y = F.conv1d(x, w, b, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups, data_format=self.data_format)

    def test_exception(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            self.dygraph_case()

class TestFunctionalConv1DErrorCase1(TestFunctionalConv1DError):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.input = np.random.randn(1, 3, 3)
        self.filter = np.random.randn(3, 3, 1)
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 0
        self.data_format = 'NCL'

class TestFunctionalConv1DErrorCase2(TestFunctionalConv1DError):

    def setUp(self):
        if False:
            return 10
        self.input = np.random.randn(0, 0, 0)
        self.filter = np.random.randn(1, 0, 0)
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.data_format = 'NCL'
if __name__ == '__main__':
    unittest.main()