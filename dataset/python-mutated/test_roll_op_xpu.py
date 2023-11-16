import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestRollOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'roll'
        self.use_dynamic_create_class = False

    class TestXPURollOp(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.op_type = 'roll'
            self.dtype = self.in_type
            self.init_shapes()
            self.inputs = {'X': np.random.random(self.x_shape).astype(self.dtype)}
            self.attrs = {'shifts': self.shifts, 'axis': self.axis}
            self.outputs = {'Out': np.roll(self.inputs['X'], self.attrs['shifts'], self.attrs['axis'])}

        def init_shapes(self):
            if False:
                return 10
            self.x_shape = (100, 4, 5)
            self.shifts = [101, -1]
            self.axis = [0, -2]

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            if False:
                print('Hello World!')
            self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Out')

    class TestRollOpCase2(TestXPURollOp):

        def init_shapes(self):
            if False:
                while True:
                    i = 10
            self.x_shape = (100, 10, 5)
            self.shifts = [8, -1]
            self.axis = [-1, -2]

    class TestRollOpCase3(TestXPURollOp):

        def init_shapes(self):
            if False:
                i = 10
                return i + 15
            self.x_shape = (100, 10, 5, 10, 15)
            self.shifts = [50, -1, 3]
            self.axis = [-1, -2, 1]

    class TestRollOpCase4(TestXPURollOp):

        def init_shapes(self):
            if False:
                return 10
            self.x_shape = (100, 10, 5, 10, 15)
            self.shifts = [8, -1]
            self.axis = [-1, -2]

    class TestRollOpCase5(TestXPURollOp):

        def init_shapes(self):
            if False:
                while True:
                    i = 10
            self.x_shape = (100, 10, 5, 10)
            self.shifts = [20, -1]
            self.axis = [0, -2]
support_types = get_xpu_op_support_types('roll')
for stype in support_types:
    create_test_class(globals(), XPUTestRollOp, stype)
if __name__ == '__main__':
    unittest.main()