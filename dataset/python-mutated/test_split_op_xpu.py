import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestSplitOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'split'
        self.use_dynamic_create_class = False

    class TestSplitOp(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.init_dtype()
            self.__class__.use_xpu = True
            self.__class__.op_type = 'split'
            self.use_mkldnn = False
            self.initParameters()
            self.inputs = {'X': self.x}
            self.attrs = {'axis': self.axis, 'sections': self.sections, 'num': self.num}
            out = np.split(self.x, self.indices_or_sections, self.axis)
            self.outputs = {'Out': [('out%d' % i, out[i]) for i in range(len(out))]}

        def init_dtype(self):
            if False:
                return 10
            self.dtype = self.in_type

        def initParameters(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = np.random.random((4, 5, 6)).astype(self.dtype)
            self.axis = 2
            self.sections = []
            self.num = 3
            self.indices_or_sections = 3

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(paddle.XPUPlace(0))

    class TestSplitOp1(TestSplitOp):

        def initParameters(self):
            if False:
                return 10
            self.x = np.random.random((4, 5, 6)).astype(self.dtype)
            self.axis = 2
            self.sections = [2, 1, -1]
            self.num = 0
            self.indices_or_sections = [2, 3]

    class TestSplitOp2(TestSplitOp):

        def initParameters(self):
            if False:
                return 10
            self.x = np.random.random((4, 5, 6)).astype(np.int32)
            self.axis = 2
            self.sections = []
            self.num = 3
            self.indices_or_sections = 3
support_types = get_xpu_op_support_types('split')
for stype in support_types:
    create_test_class(globals(), XPUTestSplitOp, stype)
if __name__ == '__main__':
    unittest.main()