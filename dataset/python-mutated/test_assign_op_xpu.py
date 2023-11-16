import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestAssignOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'assign'
        self.use_dynamic_create_class = False

    class TestAssignOPBase(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.set_case()

        def set_case(self):
            if False:
                print('Hello World!')
            self.op_type = 'assign'
            self.init_config()
            x = np.random.random(size=self.input_shape).astype(self.dtype)
            self.inputs = {'X': x}
            self.attrs = {}
            self.outputs = {'Out': x}

        def init_dtype(self):
            if False:
                return 10
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_config(self):
            if False:
                print('Hello World!')
            self.input_shape = (2, 5)

    class XPUTestAssign1(TestAssignOPBase):

        def init_config(self):
            if False:
                i = 10
                return i + 15
            self.input_shape = [2, 768]

    class XPUTestAssign3(TestAssignOPBase):

        def init_config(self):
            if False:
                while True:
                    i = 10
            self.input_shape = [1024]

    class XPUTestAssign4(TestAssignOPBase):

        def init_config(self):
            if False:
                for i in range(10):
                    print('nop')
            self.input_shape = [2, 2, 255]
support_types = get_xpu_op_support_types('assign')
for stype in support_types:
    create_test_class(globals(), XPUTestAssignOP, stype)
if __name__ == '__main__':
    unittest.main()