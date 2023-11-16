import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()
import random

class XPUTestElementwiseModOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'elementwise_floordiv'
        self.use_dynamic_create_class = False

    class TestElementwiseModOp(XPUOpTest):

        def init_kernel_type(self):
            if False:
                i = 10
                return i + 15
            self.use_mkldnn = False

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.op_type = 'elementwise_floordiv'
            self.dtype = self.in_type
            self.axis = -1
            self.init_input_output()
            self.init_kernel_type()
            self.init_axis()
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x), 'Y': OpTest.np_dtype_to_base_dtype(self.y)}
            self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
            self.outputs = {'Out': self.out}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def init_input_output(self):
            if False:
                print('Hello World!')
            self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
            self.y = np.random.uniform(1, 1000, [10, 10]).astype(self.dtype)
            self.out = np.floor_divide(self.x, self.y)

        def init_axis(self):
            if False:
                print('Hello World!')
            pass

    class TestElementwiseModOp_scalar(TestElementwiseModOp):

        def init_input_output(self):
            if False:
                print('Hello World!')
            scale_x = random.randint(0, 100000)
            scale_y = random.randint(1, 100000)
            self.x = (np.random.rand(2, 3, 4) * scale_x).astype(self.dtype)
            self.y = (np.random.rand(1) * scale_y + 1).astype(self.dtype)
            self.out = np.floor_divide(self.x, self.y)

    class TestElementwiseModOpInverse(TestElementwiseModOp):

        def init_input_output(self):
            if False:
                return 10
            self.x = np.random.uniform(0, 10000, [10]).astype(self.dtype)
            self.y = np.random.uniform(1, 1000, [10, 10]).astype(self.dtype)
            self.out = np.floor_divide(self.x, self.y)
support_types = get_xpu_op_support_types('elementwise_floordiv')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseModOp, stype)
if __name__ == '__main__':
    unittest.main()