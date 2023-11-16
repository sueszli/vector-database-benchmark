import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestSizeOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'size'
        self.use_dynamic_create_class = False

    class TestXPUSizeOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.op_type = 'size'
            self.initTestCase()
            x = np.random.random(self.shape).astype(self.dtype)
            self.inputs = {'Input': x}
            self.outputs = {'Out': np.array(np.size(x))}

        def initTestCase(self):
            if False:
                return 10
            self.shape = (6, 56, 8, 55)

        def init_dtype(self):
            if False:
                while True:
                    i = 10
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output_with_place(self.place)

    class TestSize1(TestXPUSizeOp):

        def initTestCase(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = (11, 66)

    class TestSize2(TestXPUSizeOp):

        def initTestCase(self):
            if False:
                return 10
            self.shape = (0,)

    class TestSize3(TestXPUSizeOp):

        def initTestCase(self):
            if False:
                i = 10
                return i + 15
            self.shape = (2, 3, 4, 5, 6)

    class TestSize4(TestXPUSizeOp):

        def initTestCase(self):
            if False:
                return 10
            self.shape = (12, 24)

    class TestSize5(TestXPUSizeOp):

        def initTestCase(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = (1, 64, 16)
support_types = get_xpu_op_support_types('size')
for stype in support_types:
    create_test_class(globals(), XPUTestSizeOP, stype)
if __name__ == '__main__':
    unittest.main()