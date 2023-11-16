import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestRangeOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'range'
        self.use_dynamic_create_class = False

    class TestRangeOp(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.set_xpu()
            self.op_type = 'range'
            self.init_dtype()
            self.init_config()
            self.inputs = {'Start': np.array([self.case[0]]).astype(self.dtype), 'End': np.array([self.case[1]]).astype(self.dtype), 'Step': np.array([self.case[2]]).astype(self.dtype)}
            self.outputs = {'Out': np.arange(self.case[0], self.case[1], self.case[2]).astype(self.dtype)}

        def set_xpu(self):
            if False:
                i = 10
                return i + 15
            self.__class__.no_need_check_grad = True

        def init_dtype(self):
            if False:
                return 10
            self.dtype = self.in_type

        def init_config(self):
            if False:
                print('Hello World!')
            self.case = (0, 1, 0.2) if self.dtype == np.float32 else (0, 5, 1)

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, check_dygraph=False)

    class TestRangeOpCase0(TestRangeOp):

        def init_config(self):
            if False:
                return 10
            self.case = (0, 5, 1)

    class TestRangeOpCase1(TestRangeOp):

        def init_config(self):
            if False:
                print('Hello World!')
            self.case = (0, 5, 2)

    class TestRangeOpCase2(TestRangeOp):

        def init_config(self):
            if False:
                return 10
            self.case = (10, 1, -2)

    class TestRangeOpCase3(TestRangeOp):

        def init_config(self):
            if False:
                return 10
            self.case = (-1, -10, -2)

    class TestRangeOpCase4(TestRangeOp):

        def init_config(self):
            if False:
                return 10
            self.case = (10, -10, -11)
support_types = get_xpu_op_support_types('range')
for stype in support_types:
    create_test_class(globals(), XPUTestRangeOp, stype)
if __name__ == '__main__':
    unittest.main()