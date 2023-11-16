import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle

class XPUTestClipByNormOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'clip_by_norm'
        self.use_dynamic_create_class = False

    class TestClipByNormOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'clip_by_norm'
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.use_xpu = True
            self.max_relative_error = 0.006
            self.initTestCase()
            input = np.random.random(self.shape).astype(self.dtype)
            input[np.abs(input) < self.max_relative_error] = 0.5
            self.inputs = {'X': input}
            self.attrs = {}
            self.attrs['max_norm'] = self.max_norm
            norm = np.sqrt(np.sum(np.square(input)))
            if norm > self.max_norm:
                output = self.max_norm * input / norm
            else:
                output = input
            self.outputs = {'Out': output}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            if paddle.is_compiled_with_xpu():
                paddle.enable_static()
                self.check_output_with_place(self.place)

        def initTestCase(self):
            if False:
                i = 10
                return i + 15
            self.shape = (100,)
            self.max_norm = 1.0

    class TestCase1(TestClipByNormOp):

        def initTestCase(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = (100,)
            self.max_norm = 1e+20

    class TestCase2(TestClipByNormOp):

        def initTestCase(self):
            if False:
                print('Hello World!')
            self.shape = (16, 16)
            self.max_norm = 0.1

    class TestCase3(TestClipByNormOp):

        def initTestCase(self):
            if False:
                return 10
            self.shape = (4, 8, 16)
            self.max_norm = 1.0
support_types = get_xpu_op_support_types('clip_by_norm')
for stype in support_types:
    create_test_class(globals(), XPUTestClipByNormOp, stype)
if __name__ == '__main__':
    unittest.main()