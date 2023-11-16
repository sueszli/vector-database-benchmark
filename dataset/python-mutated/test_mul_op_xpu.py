import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types

class XPUTestMulOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'mul'
        self.use_dynamic_create_class = False

    class TestXPUMulOp1(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.op_type = 'mul'
            self.dtype = self.in_type
            self.inputs = {'X': np.random.random((3, 4, 2, 9)).astype(self.in_type_str), 'Y': np.random.random((3, 6, 1, 2, 3)).astype(self.in_type_str)}
            self.attrs = {'x_num_col_dims': 2, 'y_num_col_dims': 2}
            result = np.dot(self.inputs['X'].reshape(3 * 4, 2 * 9), self.inputs['Y'].reshape(3 * 6, 1 * 2 * 3))
            result = result.reshape(3, 4, 1, 2, 3)
            self.outputs = {'Out': result}

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, atol=0.01)

        def test_check_grad_normal(self):
            if False:
                while True:
                    i = 10
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(place, ['X', 'Y'], 'Out', max_relative_error=0.1)

        def test_check_grad_ingore_x(self):
            if False:
                while True:
                    i = 10
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(place, ['Y'], 'Out', max_relative_error=0.1, no_grad_set=set('X'))

        def test_check_grad_ignore_y(self):
            if False:
                print('Hello World!')
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=0.1, no_grad_set=set('Y'))

    class TestXPUMulOp2(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.op_type = 'mul'
            self.use_xpu = True
            self.dtype = self.in_type
            self.inputs = {'X': np.random.random((20, 5)).astype(self.in_type_str), 'Y': np.random.random((5, 21)).astype(self.in_type_str)}
            self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_output_with_place(place, atol=0.01)

        def test_check_grad_normal(self):
            if False:
                return 10
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(place, ['X', 'Y'], 'Out', max_relative_error=0.1)

        def test_check_grad_ingore_x(self):
            if False:
                for i in range(10):
                    print('nop')
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(place, ['Y'], 'Out', max_relative_error=0.1, no_grad_set=set('X'))

        def test_check_grad_ingore_y(self):
            if False:
                print('Hello World!')
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=0.1, no_grad_set=set('Y'))
support_types = get_xpu_op_support_types('mul')
for stype in support_types:
    create_test_class(globals(), XPUTestMulOp, stype)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()