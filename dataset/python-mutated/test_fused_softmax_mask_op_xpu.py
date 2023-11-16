import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def _get_softmax(x, mask):
    if False:
        for i in range(10):
            print('nop')
    masked_x = (x + mask).astype('float32')
    max_value = np.max(masked_x, axis=-1, keepdims=True)
    before_exp = masked_x - max_value
    exp = np.exp(before_exp)
    exp_sum = np.sum(exp, axis=-1, keepdims=True)
    rst = exp / exp_sum
    return rst

class XPUTestFusedSoftmaxMaskOp(XPUOpTestWrapper):
    """Test fused_softmax_mask op"""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'fused_softmax_mask'
        self.use_dynamic_create_class = False

    class TestFusedSoftmaxMaskOp(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.set_xpu()
            self.op_type = 'fused_softmax_mask'
            self.init_dtype()
            self.set_input()
            self.set_output()

        def set_input(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = (1, 4, 4096, 4096)
            self.mask_shape = (1, 1, 4096, 4096)

        def set_output(self):
            if False:
                i = 10
                return i + 15
            x = np.random.random(self.x_shape).astype('float32')
            mask_input = np.random.random(self.mask_shape).astype('float32')
            self.inputs = {'X': x, 'Mask': mask_input}
            rst = _get_softmax(x, mask_input)
            self.outputs = {'Out': rst}

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def set_xpu(self):
            if False:
                return 10
            self.__class__.use_xpu = True
            self.place = paddle.XPUPlace(0)

        def init_dtype(self):
            if False:
                while True:
                    i = 10
            self.dtype = self.in_type

    class TestFusedSoftmaxMaskOp_1(TestFusedSoftmaxMaskOp):

        def set_input(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = (2, 4, 1024, 1024)
            self.mask_shape = (2, 1, 1024, 1024)

    class TestFusedSoftmaxMaskOp_2(TestFusedSoftmaxMaskOp):

        def set_input(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = (1, 4, 8192, 8192)
            self.mask_shape = (1, 1, 8192, 8192)
support_types = get_xpu_op_support_types('fused_softmax_mask')
for stype in support_types:
    create_test_class(globals(), XPUTestFusedSoftmaxMaskOp, stype)
if __name__ == '__main__':
    unittest.main()