import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def bce_loss(input, label):
    if False:
        i = 10
        return i + 15
    return -1 * (label * np.log(input) + (1.0 - label) * np.log(1.0 - input))

class XPUTestBceLossOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'bce_loss'
        self.use_dynamic_create_class = False

    class TestBceLossOp(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.op_type = 'bce_loss'
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.init_test_case()
            input_np = np.random.uniform(0.1, 0.8, self.shape).astype(self.dtype)
            label_np = np.random.randint(0, 2, self.shape).astype(self.dtype)
            output_np = bce_loss(input_np, label_np)
            self.inputs = {'X': input_np, 'Label': label_np}
            self.outputs = {'Out': output_np}

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                print('Hello World!')
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.shape = [10, 10]

    class TestBceLossOpCase1(TestBceLossOp):

        def init_test_cast(self):
            if False:
                print('Hello World!')
            self.shape = [2, 3, 4, 5]

    class TestBceLossOpCase2(TestBceLossOp):

        def init_test_cast(self):
            if False:
                print('Hello World!')
            self.shape = [2, 3, 20]
support_types = get_xpu_op_support_types('bce_loss')
for stype in support_types:
    create_test_class(globals(), XPUTestBceLossOp, stype)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()