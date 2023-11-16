import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestFillAnyLikeOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'fill_any_like'
        self.use_dynamic_create_class = False

    class TestFillAnyLikeOp(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.init_dtype()
            self.set_xpu()
            self.op_type = 'fill_any_like'
            self.place = paddle.XPUPlace(0)
            self.set_value()
            self.set_input()
            self.attrs = {'value': self.value, 'use_xpu': True}
            self.outputs = {'Out': self.value * np.ones_like(self.inputs['X'])}

        def init_dtype(self):
            if False:
                while True:
                    i = 10
            self.dtype = self.in_type

        def set_xpu(self):
            if False:
                while True:
                    i = 10
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def set_input(self):
            if False:
                return 10
            self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}

        def set_value(self):
            if False:
                while True:
                    i = 10
            self.value = 0.0

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

    class TestFillAnyLikeOp2(TestFillAnyLikeOp):

        def set_value(self):
            if False:
                print('Hello World!')
            self.value = -0.0

    class TestFillAnyLikeOp3(TestFillAnyLikeOp):

        def set_value(self):
            if False:
                print('Hello World!')
            self.value = 1.0

    class TestFillAnyLikeOp4(TestFillAnyLikeOp):

        def init(self):
            if False:
                print('Hello World!')
            self.value = 1e-09

    class TestFillAnyLikeOp5(TestFillAnyLikeOp):

        def set_value(self):
            if False:
                while True:
                    i = 10
            if self.dtype == 'float16':
                self.value = 0.05
            elif self.dtype == np.bool_:
                self.value = 1.0
            else:
                self.value = 5.0
support_types = get_xpu_op_support_types('fill_any_like')
for stype in support_types:
    create_test_class(globals(), XPUTestFillAnyLikeOp, stype)
if __name__ == '__main__':
    unittest.main()