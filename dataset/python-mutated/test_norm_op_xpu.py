import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def l2_norm(x, axis, epsilon):
    if False:
        return 10
    x2 = x ** 2
    s = np.sum(x2, axis=axis, keepdims=True)
    r = np.sqrt(s + epsilon)
    y = x / np.broadcast_to(r, x.shape)
    return (y, r)

class XPUTestNormOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'norm'
        self.use_dynamic_create_class = False

    class TestXPUNormOp(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'norm'
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.init_test_case()
            x = np.random.random(self.shape).astype(self.dtype)
            (y, norm) = l2_norm(x, self.axis, self.epsilon)
            self.inputs = {'X': x}
            self.attrs = {'epsilon': self.epsilon, 'axis': self.axis}
            self.outputs = {'Out': y, 'Norm': norm}

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.shape = [2, 3, 4, 5]
            self.axis = 1
            self.epsilon = 1e-08

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestXPUNormOp2(TestXPUNormOp):

        def init_test_case(self):
            if False:
                return 10
            self.shape = [5, 3, 9, 7]
            self.axis = 0
            self.epsilon = 1e-08

    class TestXPUNormOp3(TestXPUNormOp):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.shape = [5, 3, 2, 7]
            self.axis = -1
            self.epsilon = 1e-08

    class TestXPUNormOp4(TestXPUNormOp):

        def init_test_case(self):
            if False:
                i = 10
                return i + 15
            self.shape = [128, 1024, 14, 14]
            self.axis = 2
            self.epsilon = 1e-08

    class TestXPUNormOp5(TestXPUNormOp):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.shape = [2048, 2048]
            self.axis = 1
            self.epsilon = 1e-08
support_types = get_xpu_op_support_types('norm')
for stype in support_types:
    create_test_class(globals(), XPUTestNormOp, stype)
if __name__ == '__main__':
    unittest.main()