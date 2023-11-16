import unittest
import numpy as np
import paddle
paddle.enable_static()
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest

class XPUTestAtanOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'atan'
        self.use_dynamic_create_class = False

    class TestAtanOp(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.set_xpu()
            self.op_type = 'atan'
            self.init_input_shape()
            x = np.random.random(self.x_shape).astype(self.in_type)
            y = np.arctan(x)
            self.inputs = {'X': x}
            self.outputs = {'Out': y}

        def init_input_shape(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = [15, 6]

        def set_xpu(self):
            if False:
                print('Hello World!')
            self.__class__.no_need_check_grad = False
            self.place = paddle.XPUPlace(0)

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class Test1x1(TestAtanOp):

        def init_input_shape(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = [1, 1]

    class Test1(TestAtanOp):

        def init_input_shape(self):
            if False:
                i = 10
                return i + 15
            self.x_shape = [1]
support_types = get_xpu_op_support_types('atan')
for stype in support_types:
    create_test_class(globals(), XPUTestAtanOp, stype)
if __name__ == '__main__':
    unittest.main()