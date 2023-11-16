import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import skip_check_grad_ci
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestElementwiseMinOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'elementwise_min'
        self.use_dynamic_create_class = False

    class TestElementwiseOp(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'elementwise_min'
            self.dtype = self.in_type
            self.init_input_output()

        def init_input_output(self):
            if False:
                return 10
            x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            sgn = np.random.choice([-1, 1], [13, 17]).astype(self.dtype)
            y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad_normal(self):
            if False:
                return 10
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X', 'Y'], 'Out')

        def test_check_grad_ingore_x(self):
            if False:
                return 10
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set('X'))

        def test_check_grad_ingore_y(self):
            if False:
                for i in range(10):
                    print('nop')
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))

    @skip_check_grad_ci(reason='[skip shape check] Use y_shape(1) to test broadcast.')
    class TestElementwiseMinOp_scalar(TestElementwiseOp):

        def init_input_output(self):
            if False:
                print('Hello World!')
            x = np.random.random_integers(-5, 5, [10, 3, 4]).astype(self.dtype)
            y = np.array([0.5]).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}

    class TestElementwiseMinOp_Vector(TestElementwiseOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            x = np.random.random((100,)).astype(self.dtype)
            sgn = np.random.choice([-1, 1], (100,)).astype(self.dtype)
            y = x + sgn * np.random.uniform(0.1, 1, (100,)).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}

    class TestElementwiseMinOp_broadcast_2(TestElementwiseOp):

        def init_input_output(self):
            if False:
                print('Hello World!')
            x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(self.dtype)
            sgn = np.random.choice([-1, 1], (100,)).astype(self.dtype)
            y = x[0, 0, :] + sgn * np.random.uniform(1, 2, (100,)).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100))}

    class TestElementwiseMinOp_broadcast_4(TestElementwiseOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            x = np.random.uniform(0.5, 1, (2, 10, 2, 5)).astype(self.dtype)
            sgn = np.random.choice([-1, 1], (2, 10, 1, 5)).astype(self.dtype)
            y = x + sgn * np.random.uniform(1, 2, (2, 10, 1, 5)).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}
support_types = get_xpu_op_support_types('elementwise_min')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseMinOp, stype)
if __name__ == '__main__':
    unittest.main()