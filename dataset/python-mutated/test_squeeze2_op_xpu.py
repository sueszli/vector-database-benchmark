import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestSqueeze2Op(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'squeeze2'
        self.use_dynamic_create_class = False

    class TestSqueeze2Op(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'squeeze2'
            self.__class__.op_type = 'squeeze2'
            self.use_mkldnn = False
            self.init_dtype()
            self.init_test_case()
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'].reshape(self.new_shape), 'XShape': np.random.random(self.ori_shape).astype(self.dtype)}
            self.init_attrs()

        def init_dtype(self):
            if False:
                return 10
            self.dtype = self.in_type

        def init_attrs(self):
            if False:
                i = 10
                return i + 15
            self.attrs = {'axes': self.axes}

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.ori_shape = (1, 3, 1, 40)
            self.axes = (0, 2)
            self.new_shape = (3, 40)

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            place = paddle.XPUPlace(0)
            if self.dtype == np.bool_:
                return
            else:
                self.check_grad_with_place(place, ['X'], 'Out')

    class TestSqueeze2Op1(TestSqueeze2Op):

        def init_test_case(self):
            if False:
                return 10
            self.ori_shape = (1, 20, 1, 5)
            self.axes = (0, -2)
            self.new_shape = (20, 5)

    class TestSqueeze2Op2(TestSqueeze2Op):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.ori_shape = (1, 20, 1, 5)
            self.axes = ()
            self.new_shape = (20, 5)

    class TestSqueeze2Op3(TestSqueeze2Op):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.ori_shape = (6, 1, 5, 1, 4, 1)
            self.axes = (1, -1)
            self.new_shape = (6, 5, 1, 4)
support_types = get_xpu_op_support_types('squeeze2')
for stype in support_types:
    create_test_class(globals(), XPUTestSqueeze2Op, stype)
if __name__ == '__main__':
    unittest.main()