import unittest
import numpy as np
from test_conv3d_op import TestCase1, TestConv3DOp, TestWith1x1, TestWithGroup1, TestWithGroup2, TestWithInput1x1Filter1x1

class TestMKLDNN(TestConv3DOp):

    def init_kernel_type(self):
        if False:
            i = 10
            return i + 15
        self.use_mkldnn = True
        self.data_format = 'NCHW'
        self.dtype = np.float32

class TestMKLDNNCase1(TestCase1):

    def init_kernel_type(self):
        if False:
            return 10
        self.use_mkldnn = True
        self.data_format = 'NCHW'
        self.dtype = np.float32

class TestMKLDNNGroup1(TestWithGroup1):

    def init_kernel_type(self):
        if False:
            return 10
        self.use_mkldnn = True
        self.data_format = 'NCHW'
        self.dtype = np.float32

class TestMKLDNNGroup2(TestWithGroup2):

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True
        self.data_format = 'NCHW'
        self.dtype = np.float32

class TestMKLDNNWith1x1(TestWith1x1):

    def init_kernel_type(self):
        if False:
            while True:
                i = 10
        self.use_mkldnn = True
        self.data_format = 'NCHW'
        self.dtype = np.float32

class TestMKLDNNWithInput1x1Filter1x1(TestWithInput1x1Filter1x1):

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True
        self.data_format = 'NCHW'
        self.dtype = np.float32

class TestConv3DOp_AsyPadding_MKLDNN(TestConv3DOp):

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True
        self.data_format = 'NCHW'
        self.dtype = np.float32

    def init_paddings(self):
        if False:
            return 10
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = 'EXPLICIT'

class TestConv3DOp_Same_MKLDNN(TestConv3DOp_AsyPadding_MKLDNN):

    def init_paddings(self):
        if False:
            return 10
        self.pad = [0, 0, 0]
        self.padding_algorithm = 'SAME'

    def init_kernel_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_mkldnn = True
        self.data_format = 'NCHW'
        self.dtype = np.float32

class TestConv3DOp_Valid_MKLDNN(TestConv3DOp_AsyPadding_MKLDNN):

    def init_paddings(self):
        if False:
            for i in range(10):
                print('nop')
        self.pad = [1, 1, 1]
        self.padding_algorithm = 'VALID'

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True
        self.data_format = 'NCHW'
        self.dtype = np.float32
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()