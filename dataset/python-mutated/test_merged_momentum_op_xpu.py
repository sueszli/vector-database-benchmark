import unittest
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from test_merged_momentum_op_xpu_base import TestMergedMomentumBase
import paddle
paddle.enable_static()

class XPUTestMergedMomentumOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'merged_momentum'
        self.use_dynamic_create_class = False

    class TestMergedMomentumOp(TestMergedMomentumBase):

        def setUp(self):
            if False:
                print('Hello World!')
            super().setUp()
            self.set_case()

        def set_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shapes = [[3, 4], [2, 7], [5, 6, 8]]
            self.place = paddle.base.XPUPlace(0)
            self.seed = 1

        def testalltype(self):
            if False:
                print('Hello World!')
            self.check_with_place(self.place, self.in_type)

    class TestMergedMomentum1(TestMergedMomentumOp):

        def set_case(self):
            if False:
                while True:
                    i = 10
            self.shapes = [[3, 4], [2, 7], [5, 6, 8]]

    class TestMergedMomentum2(TestMergedMomentumOp):

        def set_case(self):
            if False:
                return 10
            self.shapes = [[3, 4], [2, 7]]

    class TestMergedMomentum3(TestMergedMomentumOp):

        def set_case(self):
            if False:
                while True:
                    i = 10
            self.shapes = [[3, 4]]

    class TestMergedMomentum4(TestMergedMomentumOp):

        def set_case(self):
            if False:
                print('Hello World!')
            self.shapes = [[3, 4], [2, 7], [5, 6, 7], [9, 9], [10, 12]]
support_types = get_xpu_op_support_types('merged_momentum')
for stype in support_types:
    create_test_class(globals(), XPUTestMergedMomentumOP, stype)
if __name__ == '__main__':
    unittest.main()