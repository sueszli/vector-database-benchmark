import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()
np.random.seed(10)

class XPUTestIsNANOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'isnan_v2'
        self.use_dynamic_create_class = False

    class TestIsNAN(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.init_dtype()
            self.set_xpu()
            self.op_type = 'isnan_v2'
            self.place = paddle.XPUPlace(0)
            self.set_inputs()
            self.set_output()

        def init_dtype(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type

        def set_inputs(self):
            if False:
                return 10
            x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
            x[0] = np.nan
            x[-1] = np.nan
            self.inputs = {'X': x}

        def set_output(self):
            if False:
                return 10
            self.outputs = {'Out': np.isnan(self.inputs['X']).astype(bool)}

        def set_xpu(self):
            if False:
                return 10
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output_with_place(self.place)
support_types = get_xpu_op_support_types('isnan_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestIsNANOp, stype)
if __name__ == '__main__':
    unittest.main()