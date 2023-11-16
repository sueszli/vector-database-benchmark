import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestSignOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'sign'
        self.use_dynamic_create_class = False

    class TestSignOPBase(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.set_case()

        def set_case(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'sign'
            self.dtype = self.in_type
            self.init_config()
            self.x = np.random.uniform(-10, 10, self.input_shape).astype(self.dtype)
            self.inputs = {'X': self.x}
            self.outputs = {'Out': np.sign(self.x)}
            self.attrs = {'use_xpu': True}

        def init_dtype(self):
            if False:
                i = 10
                return i + 15
            self.dtype = np.float32

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                i = 10
                return i + 15
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_config(self):
            if False:
                print('Hello World!')
            self.input_shape = [864]

    class XPUTestSign1(TestSignOPBase):

        def init_config(self):
            if False:
                return 10
            self.input_shape = [2, 768]

    class XPUTestSign2(TestSignOPBase):

        def init_config(self):
            if False:
                i = 10
                return i + 15
            self.input_shape = [3, 8, 4096]

    class XPUTestSign3(TestSignOPBase):

        def init_config(self):
            if False:
                i = 10
                return i + 15
            self.input_shape = [1024]

    class XPUTestSign4(TestSignOPBase):

        def init_config(self):
            if False:
                return 10
            self.input_shape = [2, 2, 255]
support_types = get_xpu_op_support_types('sign')
for stype in support_types:
    create_test_class(globals(), XPUTestSignOP, stype)
if __name__ == '__main__':
    unittest.main()