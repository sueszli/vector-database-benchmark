import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestReduceMinOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'reduce_min'

    class XPUTestReduceMinBase(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            if False:
                i = 10
                return i + 15
            self.op_type = 'reduce_min'
            self.attrs = {'use_xpu': True, 'reduce_all': self.reduce_all, 'keep_dim': self.keep_dim, 'dim': self.axis}
            self.inputs = {'X': np.random.random(self.shape).astype('float32')}
            if self.attrs['reduce_all']:
                self.outputs = {'Out': self.inputs['X'].min()}
            else:
                self.outputs = {'Out': self.inputs['X'].min(axis=self.axis, keepdims=self.attrs['keep_dim'])}

        def init_case(self):
            if False:
                while True:
                    i = 10
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                return 10
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class XPUTestReduceMinCase1(XPUTestReduceMinBase):

        def init_case(self):
            if False:
                print('Hello World!')
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceMinCase2(XPUTestReduceMinBase):

        def init_case(self):
            if False:
                return 10
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceMinCase3(XPUTestReduceMinBase):

        def init_case(self):
            if False:
                print('Hello World!')
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = True
            self.keep_dim = False

    class XPUTestReduceMinCase4(XPUTestReduceMinBase):

        def init_case(self):
            if False:
                i = 10
                return i + 15
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceMinCase5(XPUTestReduceMinBase):

        def init_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceMinCase6(XPUTestReduceMinBase):

        def init_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = True
            self.keep_dim = False
support_types = get_xpu_op_support_types('reduce_min')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceMinOp, stype)
if __name__ == '__main__':
    unittest.main()