import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def ref_logsumexp(x, axis=None, keepdim=False, reduce_all=False):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(axis, int):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if reduce_all:
        axis = None
    out = np.log(np.exp(x).sum(axis=axis, keepdims=keepdim))
    return out

class XPUTestLogsumexp(XPUOpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'logsumexp'
        self.shape = [2, 3, 4, 5]
        self.dtype = 'float32'
        self.axis = [-1]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()
        np.random.seed(10)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_logsumexp(x, self.axis, self.keepdim, self.reduce_all)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis, 'keepdim': self.keepdim, 'reduce_all': self.reduce_all}

    def set_attrs(self):
        if False:
            print('Hello World!')
        pass

    def test_check_output(self):
        if False:
            print('Hello World!')
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestLogsumexp_shape(XPUTestLogsumexp):

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.shape = [4, 5, 6]

class TestLogsumexp_axis(XPUTestLogsumexp):

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.axis = [0, -1]

class TestLogsumexp_axis_all(XPUTestLogsumexp):

    def set_attrs(self):
        if False:
            return 10
        self.axis = [0, 1, 2, 3]

class TestLogsumexp_keepdim(XPUTestLogsumexp):

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.keepdim = True

class TestLogsumexp_reduce_all(XPUTestLogsumexp):

    def set_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.reduce_all = True
if __name__ == '__main__':
    unittest.main()