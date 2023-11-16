import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestReduceSumOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'reduce_sum'

    class XPUTestReduceSumBase(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                print('Hello World!')
            self.op_type = 'reduce_sum'
            self.attrs = {'use_xpu': True, 'reduce_all': self.reduce_all, 'keep_dim': self.keep_dim, 'dim': self.axis}
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            if self.attrs['reduce_all']:
                self.outputs = {'Out': self.inputs['X'].sum()}
            else:
                self.outputs = {'Out': self.inputs['X'].sum(axis=self.axis, keepdims=self.attrs['keep_dim'])}

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                return 10
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class XPUTestReduceSumCase1(XPUTestReduceSumBase):

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceSumCase2(XPUTestReduceSumBase):

        def init_case(self):
            if False:
                return 10
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceSumCase3(XPUTestReduceSumBase):

        def init_case(self):
            if False:
                while True:
                    i = 10
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = True
            self.keep_dim = False

    class XPUTestReduceSumCase4(XPUTestReduceSumBase):

        def init_case(self):
            if False:
                return 10
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceSumCase5(XPUTestReduceSumBase):

        def init_case(self):
            if False:
                print('Hello World!')
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceSumCase6(XPUTestReduceSumBase):

        def init_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = True
            self.keep_dim = False
support_types = get_xpu_op_support_types('reduce_sum')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceSumOp, stype)
if __name__ == '__main__':
    unittest.main()