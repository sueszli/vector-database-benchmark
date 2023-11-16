import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestTakeAlongAxis(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'take_along_axis'

    class TestXPUTakeAlongAxisOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'take_along_axis'
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.init_config()
            xnp = np.random.random(self.x_shape).astype(self.dtype)
            self.target = np.take_along_axis(xnp, self.index, self.axis)
            broadcast_shape_list = list(self.x_shape)
            broadcast_shape_list[self.axis] = self.index.shape[self.axis]
            self.broadcast_shape = tuple(broadcast_shape_list)
            self.index_broadcast = np.broadcast_to(self.index, self.broadcast_shape)
            self.inputs = {'Input': xnp, 'Index': self.index_broadcast}
            self.attrs = {'Axis': self.axis}
            self.outputs = {'Result': self.target}

        def init_config(self):
            if False:
                for i in range(10):
                    print('nop')
            self.in_type = np.float32
            self.x_shape = (1, 4, 10)
            self.index_type = np.int32
            self.index = np.array([[[0, 1, 3, 5, 6]]]).astype(self.index_type)
            self.axis = 2

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            if paddle.is_compiled_with_xpu():
                self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                print('Hello World!')
            if paddle.is_compiled_with_xpu():
                self.check_grad_with_place(self.place, ['Input'], 'Result')

    class TestCase1(TestXPUTakeAlongAxisOp):

        def init_config(self):
            if False:
                i = 10
                return i + 15
            self.in_type = np.float32
            self.x_shape = (1, 10, 100)
            self.index_type = np.int32
            self.index = np.array([[[0, 1, 3, 5, 13]]]).astype(self.index_type)
            self.axis = 2

    class TestCase2(TestXPUTakeAlongAxisOp):

        def init_config(self):
            if False:
                return 10
            self.in_type = np.float32
            self.x_shape = (1, 10, 100)
            self.index_type = np.int64
            self.index = np.array([[[0, 1, 3, 5, 13]]]).astype(self.index_type)
            self.axis = 2

    class TestCase3(TestXPUTakeAlongAxisOp):

        def init_config(self):
            if False:
                print('Hello World!')
            self.in_type = np.float16
            self.x_shape = (1, 10, 100)
            self.index_type = np.int32
            self.index = np.array([[[0, 1, 3, 5, 13]]]).astype(self.index_type)
            self.axis = 2

    class TestCase4(TestXPUTakeAlongAxisOp):

        def init_config(self):
            if False:
                print('Hello World!')
            self.in_type = np.float16
            self.x_shape = (1, 10, 100)
            self.index_type = np.int64
            self.index = np.array([[[0, 1, 3, 5, 13]]]).astype(self.index_type)
            self.axis = 2

    class TestCase5(TestXPUTakeAlongAxisOp):

        def init_config(self):
            if False:
                return 10
            self.in_type = np.float32
            self.x_shape = (1, 10, 100)
            self.index_type = np.int32
            self.index = np.array([[[0], [1], [3], [5], [8]]]).astype(self.index_type)
            self.axis = 1

class XPUTestTakeAlongAxisAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        self.shape = [3, 3]
        self.index_shape = [1, 3]
        self.index_np = np.array([[0, 1, 2]]).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = [paddle.XPUPlace(0)]
        self.axis = 0

    def test_api_static(self):
        if False:
            return 10
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            index = paddle.static.data('Index', self.index_shape, 'int64')
            out = paddle.take_along_axis(x, index, self.axis)
            exe = paddle.static.Executor(self.place[0])
            res = exe.run(feed={'X': self.x_np, 'Index': self.index_np}, fetch_list=[out])
        out_ref = np.array(np.take_along_axis(self.x_np, self.index_np, self.axis))
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.001)

    def test_api_dygraph(self):
        if False:
            print('Hello World!')
        paddle.disable_static(self.place[0])
        x_tensor = paddle.to_tensor(self.x_np)
        self.index = paddle.to_tensor(self.index_np)
        out = paddle.take_along_axis(x_tensor, self.index, self.axis)
        out_ref = np.array(np.take_along_axis(self.x_np, self.index_np, self.axis))
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()

class TestTakeAlongAxisAPICase1(XPUTestTakeAlongAxisAPI):

    def setUp(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        self.shape = [2, 2]
        self.index_shape = [4, 2]
        self.index_np = np.array([[0, 0], [1, 0], [0, 0], [1, 0]]).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = [paddle.XPUPlace(0)]
        self.axis = 0
support_types = get_xpu_op_support_types('take_along_axis')
for stype in support_types:
    create_test_class(globals(), XPUTestTakeAlongAxis, stype)
if __name__ == '__main__':
    unittest.main()