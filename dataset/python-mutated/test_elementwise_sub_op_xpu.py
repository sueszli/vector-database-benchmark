import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import skip_check_grad_ci
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestElementwiseSubOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'elementwise_sub'
        self.use_dynamic_create_class = False

    class TestElementwiseOp(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.op_type = 'elementwise_sub'
            self.use_xpu = True
            self.dtype = self.in_type
            self.init_input_output()

        def init_input_output(self):
            if False:
                i = 10
                return i + 15
            self.inputs = {'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype), 'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

        def test_check_output(self):
            if False:
                return 10
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, atol=0.001)

        def test_check_grad_normal(self):
            if False:
                i = 10
                return i + 15
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X', 'Y'], 'Out')

        def test_check_grad_ingore_x(self):
            if False:
                i = 10
                return i + 15
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set('X'))

        def test_check_grad_ingore_y(self):
            if False:
                return 10
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))

    class TestElementwiseSubOp_ZeroDim1(TestElementwiseOp):

        def init_input_output(self):
            if False:
                return 10
            self.inputs = {'X': np.random.uniform(-1, 1, []).astype(self.dtype), 'Y': np.random.uniform(-1, 1, []).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_ZeroDim2(TestElementwiseOp):

        def init_input_output(self):
            if False:
                return 10
            self.inputs = {'X': np.random.uniform(-1, 1, [13, 17]).astype(self.dtype), 'Y': np.random.uniform(-1, 1, []).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_ZeroDim3(TestElementwiseOp):

        def init_input_output(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'X': np.random.uniform(-1, 1, []).astype(self.dtype), 'Y': np.random.uniform(-1, 1, [13, 17]).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    @skip_check_grad_ci(reason='[skip shape check] Use y_shape(1) to test broadcast.')
    class TestElementwiseSubOp_scalar(TestElementwiseOp):

        def init_input_output(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'X': np.random.rand(10, 3, 4).astype(self.dtype), 'Y': np.random.rand(1).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_Vector(TestElementwiseOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = {'X': np.random.random((100,)).astype(self.dtype), 'Y': np.random.random((100,)).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_broadcast_0(TestElementwiseOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = {'X': np.random.rand(100, 3, 2).astype(self.dtype), 'Y': np.random.rand(100).astype(self.dtype)}
            self.attrs = {'axis': 0}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y'].reshape(100, 1, 1)}

    class TestElementwiseSubOp_broadcast_1(TestElementwiseOp):

        def init_input_output(self):
            if False:
                print('Hello World!')
            self.inputs = {'X': np.random.rand(2, 100, 3).astype(self.dtype), 'Y': np.random.rand(100).astype(self.dtype)}
            self.attrs = {'axis': 1}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 100, 1)}

    class TestElementwiseSubOp_broadcast_2(TestElementwiseOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = {'X': np.random.rand(2, 3, 100).astype(self.dtype), 'Y': np.random.rand(100).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 1, 100)}

    class TestElementwiseSubOp_broadcast_3(TestElementwiseOp):

        def init_input_output(self):
            if False:
                return 10
            self.inputs = {'X': np.random.rand(2, 10, 12, 3).astype(self.dtype), 'Y': np.random.rand(10, 12).astype(self.dtype)}
            self.attrs = {'axis': 1}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 10, 12, 1)}

    class TestElementwiseSubOp_broadcast_4(TestElementwiseOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = {'X': np.random.rand(2, 5, 3, 12).astype(self.dtype), 'Y': np.random.rand(2, 5, 1, 12).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_commonuse_1(TestElementwiseOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = {'X': np.random.rand(2, 3, 100).astype(self.dtype), 'Y': np.random.rand(1, 1, 100).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_commonuse_2(TestElementwiseOp):

        def init_input_output(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'X': np.random.rand(10, 3, 1, 4).astype(self.dtype), 'Y': np.random.rand(10, 1, 12, 1).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_xsize_lessthan_ysize(TestElementwiseOp):

        def init_input_output(self):
            if False:
                while True:
                    i = 10
            self.inputs = {'X': np.random.rand(10, 12).astype(self.dtype), 'Y': np.random.rand(2, 3, 10, 12).astype(self.dtype)}
            self.attrs = {'axis': 2}
            self.outputs = {'Out': self.inputs['X'].reshape(1, 1, 10, 12) - self.inputs['Y']}
support_types = get_xpu_op_support_types('elementwise_sub')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseSubOp, stype)
if __name__ == '__main__':
    unittest.main()