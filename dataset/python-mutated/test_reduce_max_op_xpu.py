import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestReduceMaxOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'reduce_max'

    class XPUTestReduceMaxBase(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'reduce_max'
            self.attrs = {'use_xpu': True, 'reduce_all': self.reduce_all, 'keep_dim': self.keep_dim, 'dim': self.axis}
            self.dtype = self.in_type
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            if self.attrs['reduce_all']:
                self.outputs = {'Out': self.inputs['X'].max()}
            else:
                self.outputs = {'Out': self.inputs['X'].max(axis=self.axis, keepdims=self.attrs['keep_dim'])}

        def init_case(self):
            if False:
                return 10
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                print('Hello World!')
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class XPUTestReduceMaxCase1(XPUTestReduceMaxBase):

        def init_case(self):
            if False:
                while True:
                    i = 10
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceMaxCase2(XPUTestReduceMaxBase):

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceMaxCase3(XPUTestReduceMaxBase):

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceMaxCase4(XPUTestReduceMaxBase):

        def init_case(self):
            if False:
                while True:
                    i = 10
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = True
            self.keep_dim = False
support_types = get_xpu_op_support_types('reduce_max')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceMaxOp, stype)
if __name__ == '__main__':
    unittest.main()