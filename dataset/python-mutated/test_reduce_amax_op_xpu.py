import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestReduceAmaxOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'reduce_amax'

    class XPUTestReduceAmaxBase(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.place = paddle.XPUPlace(0)
            self.set_case()

        def set_case(self):
            if False:
                i = 10
                return i + 15
            self.op_type = 'reduce_amax'
            self.shape = (20, 10)
            self.attrs = {'use_xpu': True, 'keep_dim': False, 'dim': (1,)}
            self.inputs = {'X': np.random.randint(0, 100, self.shape).astype('float32')}
            expect_intput = self.inputs['X']
            self.outputs = {'Out': np.amax(expect_intput, axis=self.attrs['dim'], keepdims=self.attrs['keep_dim'])}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(self.place)
support_types = get_xpu_op_support_types('reduce_amax')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceAmaxOp, stype)
if __name__ == '__main__':
    unittest.main()