import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

@skip_check_grad_ci(reason='XPU does not support grad op currently')
class XPUTestElementwisePowOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'elementwise_pow'
        self.use_dynamic_create_class = False

    class TestElementwisePowOp(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.op_type = 'elementwise_pow'
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True
            self.compute_input_output()

        def compute_input_output(self):
            if False:
                print('Hello World!')
            self.inputs = {'X': np.random.uniform(1, 2, [20, 5]).astype(self.dtype), 'Y': np.random.uniform(1, 2, [20, 5]).astype(self.dtype)}
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, check_dygraph=False)

    class TestElementwisePowOp_big_shape_1(TestElementwisePowOp):

        def compute_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = {'X': np.random.uniform(1, 2, [10, 10]).astype(self.dtype), 'Y': np.random.uniform(0.1, 1, [10, 10]).astype(self.dtype)}
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOp_big_shape_2(TestElementwisePowOp):

        def compute_input_output(self):
            if False:
                print('Hello World!')
            self.inputs = {'X': np.random.uniform(1, 2, [10, 10]).astype(self.dtype), 'Y': np.random.uniform(0.2, 2, [10, 10]).astype(self.dtype)}
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    @skip_check_grad_ci(reason='[skip shape check] Use y_shape(1) to test broadcast.')
    class TestElementwisePowOp_scalar(TestElementwisePowOp):

        def compute_input_output(self):
            if False:
                print('Hello World!')
            self.inputs = {'X': np.random.uniform(0.1, 1, [3, 3, 4]).astype(self.dtype), 'Y': np.random.uniform(0.1, 1, [1]).astype(self.dtype)}
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOp_tensor(TestElementwisePowOp):

        def compute_input_output(self):
            if False:
                return 10
            self.inputs = {'X': np.random.uniform(0.1, 1, [100]).astype(self.dtype), 'Y': np.random.uniform(1, 3, [100]).astype(self.dtype)}
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOp_broadcast_0(TestElementwisePowOp):

        def compute_input_output(self):
            if False:
                return 10
            self.inputs = {'X': np.random.uniform(0.1, 1, [2, 1, 100]).astype(self.dtype), 'Y': np.random.uniform(0.1, 1, [100]).astype(self.dtype)}
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOp_broadcast_4(TestElementwisePowOp):

        def compute_input_output(self):
            if False:
                i = 10
                return i + 15
            self.inputs = {'X': np.random.uniform(0.1, 1, [2, 10, 3, 5]).astype(self.dtype), 'Y': np.random.uniform(0.1, 1, [2, 10, 1, 5]).astype(self.dtype)}
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOpInt(OpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'elementwise_pow'
            self.inputs = {'X': np.asarray([1, 3, 6]), 'Y': np.asarray([1, 1, 1])}
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output(check_dygraph=False)
support_types = get_xpu_op_support_types('elementwise_pow')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwisePowOp, stype)
if __name__ == '__main__':
    unittest.main()