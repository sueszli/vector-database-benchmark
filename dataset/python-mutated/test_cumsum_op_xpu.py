import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestCumsumOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'cumsum'
        self.use_dynamic_create_class = False

    class TestCumsumOPBase(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.set_case()

        def set_case(self):
            if False:
                print('Hello World!')
            self.op_type = 'cumsum'
            self.init_config()
            self.data = np.random.uniform(-1.0, 1.0, self.input_shape).astype(self.dtype)
            reference_out = np.cumsum(self.data, axis=self.axis)
            self.inputs = {'X': self.data}
            self.attrs = {'use_xpu': True, 'axis': self.axis, 'flatten': True if self.axis is None else False}
            self.outputs = {'Out': reference_out}

        def init_dtype(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                return 10
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_config(self):
            if False:
                print('Hello World!')
            self.input_shape = (2, 5)
            self.axis = None

    class XPUTestCumsum1(TestCumsumOPBase):

        def init_config(self):
            if False:
                i = 10
                return i + 15
            self.input_shape = [2, 768]
            self.axis = 0

    class XPUTestCumsum2(TestCumsumOPBase):

        def init_config(self):
            if False:
                print('Hello World!')
            self.input_shape = [3, 8, 4096]
            self.axis = 1

    class XPUTestCumsum3(TestCumsumOPBase):

        def init_config(self):
            if False:
                i = 10
                return i + 15
            self.input_shape = [1024]
            self.axis = 0

    class XPUTestCumsum4(TestCumsumOPBase):

        def init_config(self):
            if False:
                print('Hello World!')
            self.input_shape = [2, 2, 255]
            self.axis = -1
support_types = get_xpu_op_support_types('cumsum')
for stype in support_types:
    create_test_class(globals(), XPUTestCumsumOP, stype)
if __name__ == '__main__':
    unittest.main()