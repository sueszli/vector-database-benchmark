import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
import paddle.nn.functional as F
paddle.enable_static()
np.random.seed(10)

def ref_log_softmax(x):
    if False:
        return 10
    shiftx = x - np.max(x)
    out = shiftx - np.log(np.exp(shiftx).sum())
    return out

def ref_log_softmax_grad(x, axis):
    if False:
        for i in range(10):
            print('nop')
    if axis < 0:
        axis += len(x.shape)
    out = np.apply_along_axis(ref_log_softmax, axis, x)
    axis_dim = x.shape[axis]
    dout = np.full_like(x, fill_value=1.0 / x.size)
    dx = dout - np.exp(out) * dout.copy().sum(axis=axis, keepdims=True).repeat(axis_dim, axis=axis)
    return dx

class XPUTestLogSoftmaxOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'log_softmax'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        if False:
            i = 10
            return i + 15
        base_class = self.TestXPULogSoftmaxOp
        classes = []
        axis_arr = [-1, 1]
        shape_arr = [[2, 3, 4, 5], [12, 10], [2, 5], [7, 7], [3, 5, 7]]
        for axis in axis_arr:
            for shape in shape_arr:
                class_name = 'XPUTestLogSoftmax_' + str(axis) + '_' + str(shape)
                attr_dict = {'axis': axis, 'shape': shape}
                classes.append([class_name, attr_dict])
        return (base_class, classes)

    class TestXPULogSoftmaxOp(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.op_type = 'log_softmax'
            self.python_api = F.log_softmax
            self.dtype = 'float32'
            self.set_attrs()
            self.use_xpu = True
            if not hasattr(self, 'axis'):
                self.shape = [2, 3, 4, 5]
                self.axis = -1
            x = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
            out = np.apply_along_axis(ref_log_softmax, self.axis, x)
            self.x_grad = ref_log_softmax_grad(x, self.axis)
            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'axis': self.axis}

        def set_attrs(self):
            if False:
                while True:
                    i = 10
            pass

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output(check_dygraph=True)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_grad(['X'], ['Out'], user_defined_grads=[self.x_grad], check_dygraph=True)
support_types = get_xpu_op_support_types('log_softmax')
for stype in support_types:
    create_test_class(globals(), XPUTestLogSoftmaxOp, stype)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()