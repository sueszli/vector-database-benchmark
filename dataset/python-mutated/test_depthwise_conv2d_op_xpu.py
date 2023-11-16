import unittest
import numpy as np
import paddle
paddle.enable_static()
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from test_conv2d_op_xpu import XPUTestConv2DOp, XPUTestConv2DOp_v2

class XPUTestDepthwiseConv2DOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'depthwise_conv2d'
        self.use_dynamic_create_class = False

    class TestDepthwiseConv(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [2, 2]
            self.input_size = [2, 12, 5, 5]
            self.groups = 12
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [12, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

    class TestDepthwiseConv2(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.input_size = [2, 12, 5, 5]
            self.groups = 12
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [12, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

    class TestDepthwiseConv3(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            if False:
                return 10
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.input_size = [2, 24, 5, 5]
            self.groups = 24
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

    class TestDepthwiseConvWithDilation(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [2, 2]
            self.input_size = [2, 24, 5, 5]
            self.groups = 24
            self.dilations = [2, 2]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

    class TestDepthwiseConvWithDilation2(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.input_size = [2, 24, 5, 5]
            self.groups = 24
            self.dilations = [2, 2]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

class XPUTestDepthwiseConv2DOp_v2(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'depthwise_conv2d'
        self.use_dynamic_create_class = False

    class TestDepthwiseConv_AsyPadding(XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            if False:
                return 10
            self.use_cuda = False
            self.stride = [2, 2]
            self.input_size = [2, 12, 5, 5]
            self.groups = 12
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [12, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [1, 1, 0, 1]
            self.padding_algorithm = 'EXPLICIT'

    class TestDepthwiseConv2_AsyPadding(XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.use_cuda = False
            self.stride = [1, 1]
            self.input_size = [2, 12, 5, 5]
            self.groups = 12
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [12, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [0, 1, 0, 2]
            self.padding_algorithm = 'EXPLICIT'

    class TestDepthwiseConv3_AsyPadding(XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.use_cuda = False
            self.stride = [1, 1]
            self.input_size = [2, 24, 5, 5]
            self.groups = 24
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

        def init_paddings(self):
            if False:
                i = 10
                return i + 15
            self.pad = [1, 1, 0, 0]
            self.padding_algorithm = 'EXPLICIT'

    class TestDepthwiseConvWithDilation_AsyPadding(XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [2, 2]
            self.input_size = [2, 24, 5, 5]
            self.groups = 24
            self.dilations = [2, 2]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

        def init_paddings(self):
            if False:
                i = 10
                return i + 15
            self.pad = [1, 1, 2, 1]
            self.padding_algorithm = 'EXPLICIT'

    class TestDepthwiseConvWithDilation2_AsyPadding(XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.use_cuda = True
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.input_size = [2, 24, 5, 5]
            self.groups = 24
            self.dilations = [2, 2]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = 'depthwise_conv2d'

        def init_paddings(self):
            if False:
                print('Hello World!')
            self.pad = [0, 1, 1, 0]
            self.padding_algorithm = 'EXPLICIT'
support_types = get_xpu_op_support_types('depthwise_conv2d')
for stype in support_types:
    create_test_class(globals(), XPUTestDepthwiseConv2DOp, stype)
    create_test_class(globals(), XPUTestDepthwiseConv2DOp_v2, stype)
if __name__ == '__main__':
    unittest.main()