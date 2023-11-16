import unittest
import numpy as np
import paddle
paddle.enable_static()
from test_conv2d_transpose_op import TestConv2DTransposeOp

class TestDepthwiseConvTranspose(TestConv2DTransposeOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 4, 4]
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 4, 4]
        self.op_type = 'depthwise_conv2d_transpose'

class TestDepthwiseConvTransposeAsymmetricPad(TestConv2DTransposeOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.pad = [1, 1, 1, 2]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 4, 4]
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 3, 3]
        self.op_type = 'depthwise_conv2d_transpose'
        self.data_format = 'NCHW'

class TestDepthwiseConvTransposeSAMEPad(TestConv2DTransposeOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 4, 4]
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 3, 3]
        self.op_type = 'depthwise_conv2d_transpose'
        self.padding_algorithm = 'SAME'

class TestDepthwiseConvTransposeVALIDPad(TestConv2DTransposeOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 4, 4]
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 3, 3]
        self.op_type = 'depthwise_conv2d_transpose'
        self.padding_algorithm = 'VALID'

class TestDepthwiseConvTranspose_NHWC_3x3kernel(TestConv2DTransposeOp):

    def init_test_case(self):
        if False:
            return 10
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 4, 4, 8]
        self.groups = 8
        assert np.mod(self.input_size[3], self.groups) == 0
        f_c = self.input_size[3] // self.groups
        self.filter_size = [self.input_size[3], f_c, 3, 3]
        self.op_type = 'depthwise_conv2d_transpose'
        self.data_format = 'NHWC'
if __name__ == '__main__':
    unittest.main()