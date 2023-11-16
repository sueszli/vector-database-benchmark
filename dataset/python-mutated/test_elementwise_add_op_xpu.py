import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
import paddle
from paddle import base
paddle.enable_static()

class XPUTestElementwiseAddOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'elementwise_add'
        self.use_dynamic_create_class = False

    class TestElementwiseAddOp(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'elementwise_add'
            self.init_dtype()
            self.init_input_output()
            self.init_axis()
            self.init_max_relative_error()
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x), 'Y': OpTest.np_dtype_to_base_dtype(self.y)}
            self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
            self.outputs = {'Out': self.out}

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad_normal(self):
            if False:
                print('Hello World!')
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X', 'Y'], 'Out', max_relative_error=self.max_relative_error)

        def test_check_grad_ingore_x(self):
            if False:
                return 10
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['Y'], 'Out', no_grad_set=set('X'), max_relative_error=self.max_relative_error)

        def test_check_grad_ingore_y(self):
            if False:
                i = 10
                return i + 15
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X'], 'Out', no_grad_set=set('Y'), max_relative_error=self.max_relative_error)

        def init_input_output(self):
            if False:
                i = 10
                return i + 15
            self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.out = np.add(self.x, self.y)

        def init_dtype(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type

        def init_axis(self):
            if False:
                while True:
                    i = 10
            self.axis = -1

        def init_max_relative_error(self):
            if False:
                while True:
                    i = 10
            self.max_relative_error = 0.006

    class TestElementwiseAddOp_ZeroDim1(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                print('Hello World!')
            self.x = np.random.uniform(-1, 1, []).astype(self.dtype)
            self.y = np.random.uniform(-1, 1, []).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_ZeroDim2(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = np.random.uniform(-1, 1, []).astype(self.dtype)
            self.y = np.random.uniform(-1, 1, [13, 17]).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_ZeroDim3(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                print('Hello World!')
            self.x = np.random.uniform(-1, 1, [13, 17]).astype(self.dtype)
            self.y = np.random.uniform(-1, 1, []).astype(self.dtype)
            self.out = self.x + self.y

    @skip_check_grad_ci(reason='[skip shape check] Use y_shape(1) to test broadcast.')
    class TestElementwiseAddOp_scalar(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                i = 10
                return i + 15
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(1).astype(self.dtype)
            self.out = self.x + self.y

    @skip_check_grad_ci(reason='[skip shape check] Use y_shape(1,1) to test broadcast.')
    class TestElementwiseAddOp_scalar2(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(1, 1).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_Vector(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                return 10
            self.x = np.random.random((100,)).astype(self.dtype)
            self.y = np.random.random((100,)).astype(self.dtype)
            self.out = np.add(self.x, self.y)

    class TestElementwiseAddOp_broadcast_0(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = np.random.rand(100, 2, 3).astype(self.dtype)
            self.y = np.random.rand(100).astype(self.dtype)
            self.out = self.x + self.y.reshape(100, 1, 1)

        def init_axis(self):
            if False:
                for i in range(10):
                    print('nop')
            self.axis = 0

    class TestElementwiseAddOp_broadcast_1(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                print('Hello World!')
            self.x = np.random.rand(2, 100, 3).astype(self.dtype)
            self.y = np.random.rand(100).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 100, 1)

        def init_axis(self):
            if False:
                for i in range(10):
                    print('nop')
            self.axis = 1

    class TestElementwiseAddOp_broadcast_2(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = np.random.rand(2, 3, 100).astype(self.dtype)
            self.y = np.random.rand(100).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 1, 100)

    class TestElementwiseAddOp_broadcast_3(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                i = 10
                return i + 15
            self.x = np.random.rand(2, 10, 12, 3).astype(self.dtype)
            self.y = np.random.rand(10, 12).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 10, 12, 1)

        def init_axis(self):
            if False:
                while True:
                    i = 10
            self.axis = 1

    class TestElementwiseAddOp_broadcast_4(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                return 10
            self.x = np.random.rand(100, 2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(100, 1).astype(self.dtype)
            self.out = self.x + self.y.reshape(100, 1, 1, 1)

        def init_axis(self):
            if False:
                while True:
                    i = 10
            self.axis = 0

    class TestElementwiseAddOp_broadcast_5(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                i = 10
                return i + 15
            self.x = np.random.rand(10, 3, 12).astype(self.dtype)
            self.y = np.random.rand(10, 1, 12).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_broadcast_6(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                while True:
                    i = 10
            self.x = np.random.rand(2, 12, 3, 5).astype(self.dtype)
            self.y = np.random.rand(2, 12, 1, 5).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_broadcast_7(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                print('Hello World!')
            self.x = np.random.rand(1, 1, 20, 5).astype(self.dtype)
            self.y = np.random.rand(20, 5, 1, 1).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_rowwise_add_0(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                i = 10
                return i + 15
            self.x = np.random.rand(2, 10, 12).astype(self.dtype)
            self.y = np.random.rand(10, 12).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 10, 12)

        def init_axis(self):
            if False:
                while True:
                    i = 10
            self.axis = 1

    @skip_check_grad_ci(reason='[skip shape check] Use y_shape(1) to test broadcast.')
    class TestElementwiseAddOp_rowwise_add_1(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                while True:
                    i = 10
            self.x = np.random.rand(100, 1).astype(self.dtype)
            self.y = np.random.rand(1).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 1)

        def init_axis(self):
            if False:
                print('Hello World!')
            self.axis = 1

    class TestElementwiseAddOp_channelwise_add(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                return 10
            self.x = np.random.rand(100, 2, 3).astype(self.dtype)
            self.y = np.random.rand(100, 1, 1).astype(self.dtype)
            self.out = self.x + self.y

        def init_axis(self):
            if False:
                for i in range(10):
                    print('nop')
            self.axis = -1

    class TestElementwiseAddOp_commonuse_add1(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                print('Hello World!')
            self.x = np.random.rand(2, 3, 100).astype(self.dtype)
            self.y = np.random.rand(1, 1, 100).astype(self.dtype)
            self.out = self.x + self.y

        def init_axis(self):
            if False:
                return 10
            self.axis = -1

    class TestElementwiseAddOp_commonuse_add2(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                return 10
            self.x = np.random.rand(10, 3, 1, 4).astype(self.dtype)
            self.y = np.random.rand(10, 1, 12, 1).astype(self.dtype)
            self.out = self.x + self.y

        def init_axis(self):
            if False:
                print('Hello World!')
            self.axis = -1

    class TestElementwiseAddOp_xsize_lessthan_ysize_add(TestElementwiseAddOp):

        def init_input_output(self):
            if False:
                while True:
                    i = 10
            self.x = np.random.rand(10, 12).astype(self.dtype)
            self.y = np.random.rand(2, 3, 10, 12).astype(self.dtype)
            self.out = self.x + self.y

        def init_axis(self):
            if False:
                return 10
            self.axis = 2

    class TestAddOp(unittest.TestCase):

        def test_name(self):
            if False:
                print('Hello World!')
            with base.program_guard(base.Program()):
                x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
                y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')
                y_1 = paddle.add(x, y, name='add_res')
                self.assertEqual('add_res' in y_1.name, True)

        def test_declarative(self):
            if False:
                while True:
                    i = 10
            with base.program_guard(base.Program()):

                def gen_data():
                    if False:
                        print('Hello World!')
                    return {'x': np.array([2, 3, 4]).astype('float32'), 'y': np.array([1, 5, 2]).astype('float32')}
                x = paddle.static.data(name='x', shape=[3], dtype='float32')
                y = paddle.static.data(name='y', shape=[3], dtype='float32')
                z = paddle.add(x, y)
                place = base.XPUPlace(0)
                exe = base.Executor(place)
                z_value = exe.run(feed=gen_data(), fetch_list=[z.name])
                z_expected = np.array([3.0, 8.0, 6.0])
                self.assertEqual((z_value == z_expected).all(), True)

        def test_dygraph(self):
            if False:
                while True:
                    i = 10
            with base.dygraph.guard():
                np_x = np.array([2, 3, 4]).astype('float32')
                np_y = np.array([1, 5, 2]).astype('float32')
                x = base.dygraph.to_variable(np_x)
                y = base.dygraph.to_variable(np_y)
                z = paddle.add(x, y)
                np_z = z.numpy()
                z_expected = np.array([3.0, 8.0, 6.0])
                self.assertEqual((np_z == z_expected).all(), True)
support_types = get_xpu_op_support_types('elementwise_add')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseAddOp, stype)
if __name__ == '__main__':
    unittest.main()