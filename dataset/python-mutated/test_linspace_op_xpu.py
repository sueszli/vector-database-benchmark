import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest, convert_np_dtype_to_dtype_
import paddle
paddle.enable_static()

class XPUTestLinspaceOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'linspace'
        self.use_dynamic_create_class = False

    class TestXPULinespaceOp(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'linspace'
            self.dtype = self.in_type
            self.set_attrs()
            self.atol = 0.0001
            np.random.seed(10)
            self.inputs = {'Start': np.array([0]).astype(self.dtype), 'Stop': np.array([10]).astype(self.dtype), 'Num': np.array([11]).astype('int32')}
            self.outputs = {'Out': np.arange(0, 11).astype(self.dtype)}
            self.attrs = {'dtype': int(convert_np_dtype_to_dtype_(self.dtype))}

        def set_attrs(self):
            if False:
                print('Hello World!')
            pass

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(paddle.XPUPlace(0), atol=self.atol)

    class TestXPULinespace2(TestXPULinespaceOp):

        def set_attrs(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'Start': np.array([10]).astype(self.dtype), 'Stop': np.array([0]).astype(self.dtype), 'Num': np.array([11]).astype('int32')}
            self.outputs = {'Out': np.arange(10, -1, -1).astype(self.dtype)}

    class TestXPULinespace3(TestXPULinespaceOp):

        def set_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = {'Start': np.array([10]).astype(self.dtype), 'Stop': np.array([0]).astype(self.dtype), 'Num': np.array([1]).astype('int32')}
            self.outputs = {'Out': np.array(10, dtype=self.dtype)}
support_types = get_xpu_op_support_types('linspace')
for stype in support_types:
    create_test_class(globals(), XPUTestLinspaceOp, stype)
if __name__ == '__main__':
    unittest.main()