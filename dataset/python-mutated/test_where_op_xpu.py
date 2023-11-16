import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest
import paddle
from paddle import base
from paddle.base.backward import append_backward
paddle.enable_static()

class XPUTestWhereOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'where'

    class TestXPUWhereOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.init_config()
            self.init_data()
            self.convert_data_if_bf16()
            self.inputs = {'Condition': self.cond, 'X': self.x, 'Y': self.y}
            self.outputs = {'Out': np.where(self.cond, self.x, self.y)}

        def init_data(self):
            if False:
                print('Hello World!')
            self.x = np.random.uniform(-3, 5, 100)
            self.y = np.random.uniform(-3, 5, 100)
            self.cond = np.zeros(100).astype('bool')

        def convert_data_if_bf16(self):
            if False:
                while True:
                    i = 10
            if self.dtype == np.uint16:
                self.x = convert_float_to_uint16(self.x)
                self.y = convert_float_to_uint16(self.y)
            else:
                self.x = self.x.astype(self.dtype)
                self.y = self.y.astype(self.dtype)

        def init_config(self):
            if False:
                print('Hello World!')
            self.op_type = 'where'
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                i = 10
                return i + 15
            self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    class TestXPUWhereOp2(TestXPUWhereOp):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.x = np.random.uniform(-5, 5, (60, 2))
            self.y = np.random.uniform(-5, 5, (60, 2))
            self.cond = np.ones((60, 2)).astype('bool')

    class TestXPUWhereOp3(TestXPUWhereOp):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.x = np.random.uniform(-3, 5, (20, 2, 4))
            self.y = np.random.uniform(-3, 5, (20, 2, 4))
            self.cond = np.array(np.random.randint(2, size=(20, 2, 4)), dtype=bool)
support_types = get_xpu_op_support_types('where')
for stype in support_types:
    create_test_class(globals(), XPUTestWhereOp, stype)

class TestXPUWhereAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.__class__.use_xpu = True
        self.place = paddle.XPUPlace(0)
        self.init_data()

    def init_data(self):
        if False:
            return 10
        self.shape = [10, 15]
        self.cond = np.array(np.random.randint(2, size=self.shape), dtype=bool)
        self.x = np.random.uniform(-2, 3, self.shape).astype(np.float32)
        self.y = np.random.uniform(-2, 3, self.shape).astype(np.float32)
        self.out = np.where(self.cond, self.x, self.y)

    def ref_x_backward(self, dout):
        if False:
            return 10
        return np.where(self.cond, dout, 0)

    def ref_y_backward(self, dout):
        if False:
            while True:
                i = 10
        return np.where(~self.cond, dout, 0)

    def test_api(self):
        if False:
            i = 10
            return i + 15
        for x_stop_gradient in [False, True]:
            for y_stop_gradient in [False, True]:
                train_prog = base.Program()
                startup = base.Program()
                with base.program_guard(train_prog, startup):
                    cond = paddle.static.data(name='cond', shape=self.shape, dtype='bool')
                    x = paddle.static.data(name='x', shape=self.shape, dtype='float32')
                    y = paddle.static.data(name='y', shape=self.shape, dtype='float32')
                    x.stop_gradient = x_stop_gradient
                    y.stop_gradient = y_stop_gradient
                    result = paddle.where(cond, x, y)
                    result.stop_gradient = False
                    append_backward(paddle.mean(result))
                    exe = base.Executor(self.place)
                    exe.run(startup)
                    fetch_list = [result, result.grad_name]
                    if x_stop_gradient is False:
                        fetch_list.append(x.grad_name)
                    if y_stop_gradient is False:
                        fetch_list.append(y.grad_name)
                    out = exe.run(train_prog, feed={'cond': self.cond, 'x': self.x, 'y': self.y}, fetch_list=fetch_list)
                    np.testing.assert_array_equal(out[0], self.out)
                    if x_stop_gradient is False:
                        np.testing.assert_array_equal(out[2], self.ref_x_backward(out[1]))
                        if y.stop_gradient is False:
                            np.testing.assert_array_equal(out[3], self.ref_y_backward(out[1]))
                    elif y.stop_gradient is False:
                        np.testing.assert_array_equal(out[2], self.ref_y_backward(out[1]))

    def test_api_broadcast(self, use_cuda=False):
        if False:
            i = 10
            return i + 15
        train_prog = base.Program()
        startup = base.Program()
        with base.program_guard(train_prog, startup):
            x = paddle.static.data(name='x', shape=[-1, 4, 1], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 4, 2], dtype='float32')
            x_i = np.array([[0.9383, 0.1983, 3.2, 1.2]]).astype('float32')
            y_i = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]).astype('float32')
            result = paddle.where(x > 1, x=x, y=y)
            exe = base.Executor(self.place)
            exe.run(startup)
            out = exe.run(train_prog, feed={'x': x_i, 'y': y_i}, fetch_list=[result])
            np.testing.assert_array_equal(out[0], np.where(x_i > 1, x_i, y_i))

class TestWhereDygraphAPI(unittest.TestCase):

    def test_api(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard(paddle.XPUPlace(0)):
            x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype('float32')
            y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype('float32')
            cond_i = np.array([False, False, True, True]).astype('bool')
            x = base.dygraph.to_variable(x_i)
            y = base.dygraph.to_variable(y_i)
            cond = base.dygraph.to_variable(cond_i)
            out = paddle.where(cond, x, y)
            np.testing.assert_array_equal(out.numpy(), np.where(cond_i, x_i, y_i))
if __name__ == '__main__':
    unittest.main()