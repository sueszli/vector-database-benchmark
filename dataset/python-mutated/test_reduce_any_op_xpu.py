import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestReduceAnyOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'reduce_any'

    class XPUTestReduceAnyBase(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.place = paddle.XPUPlace(0)
            self.set_case()

        def set_case(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'reduce_any'
            self.attrs = {'use_xpu': True, 'reduce_all': False, 'keep_dim': False, 'dim': (3, 5, 4)}
            self.inputs = {'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype('bool')}
            self.outputs = {'Out': self.inputs['X'].any(axis=self.attrs['dim'])}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

    class XPUTestReduceAnyCase1(XPUTestReduceAnyBase):

        def set_case(self):
            if False:
                return 10
            self.op_type = 'reduce_any'
            self.attrs = {'use_xpu': True, 'dim': [1]}
            self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype('bool')}
            self.outputs = {'Out': self.inputs['X'].any(axis=1)}

    class XPUTestReduceAnyCase2(XPUTestReduceAnyBase):

        def set_case(self):
            if False:
                print('Hello World!')
            self.op_type = 'reduce_any'
            self.attrs = {'use_xpu': True, 'reduce_all': False, 'keep_dim': False, 'dim': (3, 6)}
            self.inputs = {'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype('bool')}
            self.outputs = {'Out': self.inputs['X'].any(axis=self.attrs['dim'])}
support_types = get_xpu_op_support_types('reduce_any')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceAnyOp, stype)
if __name__ == '__main__':
    unittest.main()