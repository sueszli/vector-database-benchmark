import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestIncrementOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'increment'
        self.use_dynamic_create_class = False

    class TestXPUIncrementOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.op_type = 'increment'
            self.initTestCase()
            x = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            output = x + np.cast[self.dtype](self.step)
            output = output.astype(self.dtype)
            self.inputs = {'X': x}
            self.attrs = {'step': self.step}
            self.outputs = {'Out': output}

        def initTestCase(self):
            if False:
                print('Hello World!')
            self.step = -1.5

        def init_dtype(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

    class TestIncrement1(TestXPUIncrementOp):

        def initTestCase(self):
            if False:
                for i in range(10):
                    print('nop')
            self.step = 6.0

    class TestIncrement2(TestXPUIncrementOp):

        def initTestCase(self):
            if False:
                return 10
            self.step = 2.1

    class TestIncrement3(TestXPUIncrementOp):

        def initTestCase(self):
            if False:
                while True:
                    i = 10
            self.step = -1.5

    class TestIncrement4(TestXPUIncrementOp):

        def initTestCase(self):
            if False:
                i = 10
                return i + 15
            self.step = 0.5

    class TestIncrement5(TestXPUIncrementOp):

        def initTestCase(self):
            if False:
                i = 10
                return i + 15
            self.step = 3
support_types = get_xpu_op_support_types('increment')
for stype in support_types:
    create_test_class(globals(), XPUTestIncrementOP, stype)
if __name__ == '__main__':
    unittest.main()