import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()
np.random.seed(10)

def stable_softmax(x):
    if False:
        for i in range(10):
            print('nop')
    'Compute the softmax of vector x in a numerically stable way.'
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def ref_softmax(x, axis=None, dtype=None):
    if False:
        i = 10
        return i + 15
    x_t = x.copy()
    if dtype is not None:
        x_t = x_t.astype(dtype)
    if axis is None:
        axis = -1
    return np.apply_along_axis(stable_softmax, axis, x_t)

class XPUTestSoftmaxOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'softmax'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        if False:
            i = 10
            return i + 15
        base_class = self.TestSoftmaxOp
        classes = []
        shapes = [[2, 3, 4, 5], [7, 1], [63, 18], [2, 38512], [3, 4095]]
        axis = [-1, 0, 1]
        for shape in shapes:
            for axi in axis:
                class_name = 'XPUTestSoftmax_' + str(shape) + '_' + str(axi)
                attr_dict = {'shape': shape, 'axis': axi}
                classes.append([class_name, attr_dict])
        return (base_class, classes)

    class TestSoftmaxOp(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.op_type = 'softmax'
            if not hasattr(self, 'shape'):
                self.shape = [1, 7]
                self.axis = -1
            self.dtype = np.float32
            x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            out = np.apply_along_axis(stable_softmax, self.axis, x)
            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'axis': self.axis, 'use_xpu': True}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(paddle.XPUPlace(0), atol=0.0001)

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Out')
support_types = get_xpu_op_support_types('softmax')
for stype in support_types:
    create_test_class(globals(), XPUTestSoftmaxOp, stype)
if __name__ == '__main__':
    unittest.main()