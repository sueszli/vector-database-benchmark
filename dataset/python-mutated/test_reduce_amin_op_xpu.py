import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestReduceAmaxOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'reduce_amin'

    class XPUTestReduceAmaxBase(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.place = paddle.XPUPlace(0)
            self.set_case()

        def set_case(self):
            if False:
                print('Hello World!')
            self.op_type = 'reduce_amin'
            self.shape = (20, 10)
            self.attrs = {'use_xpu': True, 'keep_dim': False, 'dim': (1,)}
            self.inputs = {'X': np.random.randint(0, 100, self.shape).astype('float32')}
            expect_intput = self.inputs['X']
            self.outputs = {'Out': np.amin(expect_intput, axis=self.attrs['dim'], keepdims=self.attrs['keep_dim'])}

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)
support_types = get_xpu_op_support_types('reduce_amin')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceAmaxOp, stype)
if __name__ == '__main__':
    unittest.main()