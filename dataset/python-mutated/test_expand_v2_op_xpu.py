import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
from paddle import base
paddle.enable_static()
np.random.seed(10)

class XPUTestExpandV2Op(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'expand_v2'
        self.use_dynamic_create_class = False

    class TestExpandV2XPUOp(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.init_dtype()
            self.set_xpu()
            self.op_type = 'expand_v2'
            self.place = paddle.XPUPlace(0)
            self.init_data()
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype)}
            self.attrs = {'shape': self.shape}
            output = np.tile(self.inputs['X'], self.expand_times)
            self.outputs = {'Out': output}

        def init_dtype(self):
            if False:
                i = 10
                return i + 15
            self.dtype = self.in_type

        def set_xpu(self):
            if False:
                print('Hello World!')
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.ori_shape = [100]
            self.shape = [100]
            self.expand_times = [1]

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestExpandV2OpRank2_DimExpanding(TestExpandV2XPUOp):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = [120]
            self.shape = [2, 120]
            self.expand_times = [2, 1]

    class TestExpandV2OpRank2(TestExpandV2XPUOp):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = [1, 140]
            self.shape = [12, 140]
            self.expand_times = [12, 1]

    class TestExpandV2OpRank3_Corner(TestExpandV2XPUOp):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = (2, 10, 5)
            self.shape = (2, 10, 5)
            self.expand_times = (1, 1, 1)

    class TestExpandV2OpRank4(TestExpandV2XPUOp):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.ori_shape = (2, 4, 5, 7)
            self.shape = (-1, -1, -1, -1)
            self.expand_times = (1, 1, 1, 1)

    class TestExpandV2OpRank5(TestExpandV2XPUOp):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.ori_shape = (2, 4, 1, 15)
            self.shape = (2, -1, 4, -1)
            self.expand_times = (1, 1, 4, 1)

    class TestExpandV2OpRank6(TestExpandV2XPUOp):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = (4, 1, 30)
            self.shape = (2, -1, 4, 30)
            self.expand_times = (2, 1, 4, 1)

    class TestExpandV2OpXPURank1_tensor_attr(TestExpandV2XPUOp):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.set_xpu()
            self.place = paddle.XPUPlace(0)
            self.op_type = 'expand_v2'
            self.init_data()
            self.dtype = np.float32
            expand_shapes_tensor = []
            for (index, ele) in enumerate(self.expand_shape):
                expand_shapes_tensor.append(('x' + str(index), np.ones(1).astype('int32') * ele))
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype), 'expand_shapes_tensor': expand_shapes_tensor}
            self.attrs = {'shape': self.infer_expand_shape}
            output = np.tile(self.inputs['X'], self.expand_times)
            self.outputs = {'Out': output}

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = [100]
            self.expand_times = [1]
            self.expand_shape = [100]
            self.infer_expand_shape = [-1]

    class TestExpandV2OpRank2_Corner_tensor_attr(TestExpandV2OpXPURank1_tensor_attr):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ori_shape = [12, 14]
            self.expand_times = [1, 1]
            self.expand_shape = [12, 14]
            self.infer_expand_shape = [12, -1]

    class TestExpandV2XPUOp_tensor(TestExpandV2XPUOp):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.set_xpu()
            self.place = paddle.XPUPlace(0)
            self.op_type = 'expand_v2'
            self.init_data()
            self.dtype = np.float32
            self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype), 'Shape': np.array(self.expand_shape).astype('int32')}
            self.attrs = {}
            output = np.tile(self.inputs['X'], self.expand_times)
            self.outputs = {'Out': output}

        def init_data(self):
            if False:
                return 10
            self.ori_shape = [100]
            self.expand_times = [2, 1]
            self.expand_shape = [2, 100]

class TestExpandV2OpInteger(XPUOpTest):

    def init_type(self):
        if False:
            print('Hello World!')
        self.dtype = 'int32'

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_xpu()
        self.init_type()
        self.place = paddle.XPUPlace(0)
        self.op_type = 'expand_v2'
        self.inputs = {'X': np.random.randint(10, size=(2, 4, 20)).astype(self.dtype)}
        self.attrs = {'shape': [2, 4, 20]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def set_xpu(self):
        if False:
            i = 10
            return i + 15
        self.__class__.use_xpu = True

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad_with_place(self.place, ['X'], 'Out')

class TestExpandV2API(unittest.TestCase):

    def test_static(self):
        if False:
            print('Hello World!')
        with base.program_guard(base.Program(), base.Program()):
            input = np.random.random([12, 14]).astype('float32')
            x = paddle.static.data(name='x', shape=[12, 14], dtype='float32')
            positive_2 = paddle.tensor.fill_constant([1], 'int32', 12)
            expand_shape = paddle.static.data(name='expand_shape', shape=[2], dtype='int32')
            out_1 = paddle.expand(x, shape=[12, 14])
            out_2 = paddle.expand(x, shape=[positive_2, 14])
            out_3 = paddle.expand(x, shape=expand_shape)
            g0 = base.backward.calc_gradient(out_2, x)
            exe = base.Executor(place=paddle.XPUPlace(0))
            (res_1, res_2, res_3) = exe.run(base.default_main_program(), feed={'x': input, 'expand_shape': np.array([12, 14]).astype('int32')}, fetch_list=[out_1, out_2, out_3])
            np.testing.assert_array_equal(res_1, np.tile(input, (1, 1)))
            np.testing.assert_array_equal(res_2, np.tile(input, (1, 1)))
            np.testing.assert_array_equal(res_3, np.tile(input, (1, 1)))
support_types = get_xpu_op_support_types('expand_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestExpandV2Op, stype)
if __name__ == '__main__':
    unittest.main()