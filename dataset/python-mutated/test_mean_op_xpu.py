import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle
from paddle.base import Program, program_guard
np.random.seed(10)
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
paddle.enable_static()

class XPUTestMeanOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'mean'
        self.use_dynamic_create_class = False

    class TestMeanOp(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.init_dtype()
            self.set_xpu()
            self.op_type = 'mean'
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            self.outputs = {'Out': np.mean(self.inputs['X']).astype(np.float16)}

        def init_dtype(self):
            if False:
                print('Hello World!')
            self.dtype = self.in_type

        def set_shape(self):
            if False:
                while True:
                    i = 10
            self.shape = (10, 10)

        def set_xpu(self):
            if False:
                for i in range(10):
                    print('nop')
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.dtype

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_checkout_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestMeanOp1(TestMeanOp):

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = 5

    class TestMeanOp2(TestMeanOp):

        def set_shape(self):
            if False:
                return 10
            self.shape = (5, 7, 8)

    class TestMeanOp3(TestMeanOp):

        def set_shape(self):
            if False:
                while True:
                    i = 10
            self.shape = (10, 5, 7, 8)

    class TestMeanOp4(TestMeanOp):

        def set_shape(self):
            if False:
                return 10
            self.shape = (2, 2, 3, 3, 3)

class TestMeanOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        with program_guard(Program(), Program()):
            input1 = 12
            self.assertRaises(TypeError, paddle.mean, input1)
            input2 = paddle.static.data(name='input2', shape=[-1, 12, 10], dtype='int32')
            self.assertRaises(TypeError, paddle.mean, input2)
            input3 = paddle.static.data(name='input3', shape=[-1, 4], dtype='float16')
            paddle.nn.functional.softmax(input3)
support_types = get_xpu_op_support_types('mean')
for stype in support_types:
    create_test_class(globals(), XPUTestMeanOp, stype)
if __name__ == '__main__':
    unittest.main()