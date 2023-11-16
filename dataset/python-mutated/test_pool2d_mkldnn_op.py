import unittest
import numpy as np
from test_pool2d_op import TestCase1, TestCase2, TestCase3, TestCase4, TestCase5, TestPool2D_Op, avg_pool2D_forward_naive

def create_test_mkldnn_use_ceil_class(parent):
    if False:
        for i in range(10):
            print('nop')

    class TestMKLDNNPool2DUseCeilCase(parent):

        def init_kernel_type(self):
            if False:
                for i in range(10):
                    print('nop')
            self.use_mkldnn = True

        def init_ceil_mode(self):
            if False:
                print('Hello World!')
            self.ceil_mode = True

        def init_data_type(self):
            if False:
                print('Hello World!')
            self.dtype = np.float32
    cls_name = '{}_{}'.format(parent.__name__, 'MKLDNNCeilModeCast')
    TestMKLDNNPool2DUseCeilCase.__name__ = cls_name
    globals()[cls_name] = TestMKLDNNPool2DUseCeilCase
create_test_mkldnn_use_ceil_class(TestPool2D_Op)
create_test_mkldnn_use_ceil_class(TestCase1)
create_test_mkldnn_use_ceil_class(TestCase2)

def create_test_mkldnn_class(parent):
    if False:
        for i in range(10):
            print('nop')

    class TestMKLDNNCase(parent):

        def init_kernel_type(self):
            if False:
                print('Hello World!')
            self.use_mkldnn = True

        def init_data_type(self):
            if False:
                while True:
                    i = 10
            self.dtype = np.float32
    cls_name = '{}_{}'.format(parent.__name__, 'MKLDNNOp')
    TestMKLDNNCase.__name__ = cls_name
    globals()[cls_name] = TestMKLDNNCase
create_test_mkldnn_class(TestPool2D_Op)
create_test_mkldnn_class(TestCase1)
create_test_mkldnn_class(TestCase2)
create_test_mkldnn_class(TestCase3)
create_test_mkldnn_class(TestCase4)
create_test_mkldnn_class(TestCase5)

class TestAvgPoolAdaptive(TestPool2D_Op):

    def init_adaptive(self):
        if False:
            for i in range(10):
                print('nop')
        self.adaptive = True

    def init_pool_type(self):
        if False:
            i = 10
            return i + 15
        self.pool_type = 'avg'
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.ksize = [1, 1]
        self.strides = [1, 1]

    def init_data_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.float32

    def init_global_pool(self):
        if False:
            for i in range(10):
                print('nop')
        self.global_pool = False

class TestAvgPoolAdaptive2(TestAvgPoolAdaptive):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.ksize = [2, 3]
        self.strides = [1, 1]

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.shape = [2, 3, 6, 6]

class TestAvgPoolAdaptive3(TestAvgPoolAdaptive):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.shape = [1, 3, 16, 16]

class TestAsymPad(TestPool2D_Op):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_paddings(self):
        if False:
            while True:
                i = 10
        self.paddings = [1, 0, 1, 0]

    def init_pool_type(self):
        if False:
            i = 10
            return i + 15
        self.pool_type = 'avg'
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        if False:
            i = 10
            return i + 15
        self.global_pool = False

    def init_shape(self):
        if False:
            while True:
                i = 10
        self.shape = [2, 3, 7, 7]

    def init_kernel_type(self):
        if False:
            while True:
                i = 10
        self.use_mkldnn = True

    def init_data_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.float32

class TestAsymPadCase1(TestAsymPad):

    def init_paddings(self):
        if False:
            return 10
        self.paddings = [1, 1, 0, 0]

class TestAsymPadCase2(TestAsymPad):

    def init_paddings(self):
        if False:
            return 10
        self.paddings = [1, 0, 1, 2]

class TestAsymPadCase3(TestAsymPad):

    def init_paddings(self):
        if False:
            return 10
        self.paddings = [1, 2, 1, 2]

class TestAsymPadCase4(TestAsymPad):

    def init_paddings(self):
        if False:
            for i in range(10):
                print('nop')
        self.paddings = [1, 0, 1, 2]

class TestAsymPadCase5(TestAsymPad):

    def init_paddings(self):
        if False:
            for i in range(10):
                print('nop')
        self.paddings = [2, 2, 1, 2]

class TestAsymPadMaxCase1(TestAsymPadCase1):

    def init_pool_type(self):
        if False:
            while True:
                i = 10
        self.pool_type = 'max'

class TestAsymPadMaxCase2(TestAsymPadCase2):

    def init_pool_type(self):
        if False:
            print('Hello World!')
        self.pool_type = 'max'

class TestAsymPadMaxCase3(TestAsymPadCase3):

    def init_pool_type(self):
        if False:
            return 10
        self.pool_type = 'max'

class TestAsymPadMaxCase4(TestAsymPadCase4):

    def init_pool_type(self):
        if False:
            i = 10
            return i + 15
        self.pool_type = 'max'

class TestAsymPadMaxCase5(TestAsymPadCase5):

    def init_pool_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.pool_type = 'max'

class TestAsymPadSame(TestAsymPad):

    def init_paddings(self):
        if False:
            while True:
                i = 10
        self.paddings = [0, 0]
        self.padding_algorithm = 'SAME'

class TestAsymPadValid(TestAsymPad):

    def init_paddings(self):
        if False:
            print('Hello World!')
        self.paddings = [0, 0, 0, 0]
        self.padding_algorithm = 'VALID'

class TestAsymPadValidNHWC(TestAsymPadValid):

    def init_data_format(self):
        if False:
            i = 10
            return i + 15
        self.data_format = 'NHWC'

    def init_shape(self):
        if False:
            while True:
                i = 10
        self.shape = [2, 7, 7, 3]
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()