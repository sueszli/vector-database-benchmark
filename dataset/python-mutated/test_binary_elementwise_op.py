import unittest
import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestBinaryOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.init_case()

    def get_x_data(self):
        if False:
            i = 10
            return i + 15
        return self.random([32, 64], 'float32', -10.0, 10.0)

    def get_y_data(self):
        if False:
            return 10
        return self.random([32, 64], 'float32', -10.0, 10.0)

    def get_axis_value(self):
        if False:
            for i in range(10):
                print('nop')
        return -1

    def init_case(self):
        if False:
            while True:
                i = 10
        self.inputs = {'x': self.get_x_data(), 'y': self.get_y_data()}
        self.axis = self.get_axis_value()

    def paddle_func(self, x, y):
        if False:
            while True:
                i = 10
        return paddle.add(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            print('Hello World!')
        return builder.add(x, y, axis)

    def build_paddle_program(self, target):
        if False:
            while True:
                i = 10
        x = paddle.to_tensor(self.inputs['x'], stop_gradient=False)
        y = paddle.to_tensor(self.inputs['y'], stop_gradient=False)

        def get_unsqueeze_axis(x_rank, y_rank, axis):
            if False:
                while True:
                    i = 10
            self.assertTrue(x_rank >= y_rank, 'The rank of x should be greater or equal to that of y.')
            axis = axis if axis >= 0 else x_rank - y_rank
            unsqueeze_axis = np.arange(0, axis).tolist() + np.arange(axis + y_rank, x_rank).tolist()
            return unsqueeze_axis
        unsqueeze_axis = get_unsqueeze_axis(len(self.inputs['x'].shape), len(self.inputs['y'].shape), self.axis)
        y_t = paddle.unsqueeze(y, axis=unsqueeze_axis) if len(unsqueeze_axis) > 0 else y
        out = self.paddle_func(x, y_t)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        if False:
            return 10
        builder = NetBuilder('binary_elementwise_test')
        x = builder.create_input(self.nptype2cinntype(self.inputs['x'].dtype), self.inputs['x'].shape, 'x')
        y = builder.create_input(self.nptype2cinntype(self.inputs['y'].dtype), self.inputs['y'].shape, 'y')
        out = self.cinn_func(builder, x, y, axis=self.axis)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, y], [self.inputs['x'], self.inputs['y']], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        if False:
            i = 10
            return i + 15
        self.check_outputs_and_grads()

class TestAddOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            return 10
        return paddle.add(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            print('Hello World!')
        return builder.add(x, y, axis)

class TestAddOpFP64(TestAddOp):

    def get_x_data(self):
        if False:
            print('Hello World!')
        return self.random([32, 64], 'float64', -10.0, 10.0)

    def get_y_data(self):
        if False:
            while True:
                i = 10
        return self.random([32, 64], 'float64', -10.0, 10.0)

class TestAddOpFP16(TestAddOp):

    def get_x_data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.random([32, 64], 'float16', -10.0, 10.0)

    def get_y_data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.random([32, 64], 'float16', -10.0, 10.0)

class TestAddOpInt32(TestAddOp):

    def get_x_data(self):
        if False:
            return 10
        return self.random([32, 64], 'int32', -10.0, 10.0)

    def get_y_data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.random([32, 64], 'int32', -10.0, 10.0)

class TestAddOpInt64(TestAddOp):

    def get_x_data(self):
        if False:
            while True:
                i = 10
        return self.random([32, 64], 'int64', -10.0, 10.0)

    def get_y_data(self):
        if False:
            while True:
                i = 10
        return self.random([32, 64], 'int64', -10.0, 10.0)

class TestSubtractOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            while True:
                i = 10
        return paddle.subtract(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            print('Hello World!')
        return builder.subtract(x, y, axis)

class TestDivideOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        return paddle.divide(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            return 10
        return builder.divide(x, y, axis)

class TestMultiplyOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            print('Hello World!')
        return paddle.multiply(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            print('Hello World!')
        return builder.multiply(x, y, axis)

class TestFloorDivideOp(TestBinaryOp):

    def get_x_data(self):
        if False:
            i = 10
            return i + 15
        return self.random([32, 64], 'int32', 1, 100) * np.random.choice([-1, 1], [1])[0]

    def get_y_data(self):
        if False:
            while True:
                i = 10
        return self.random([32, 64], 'int32', 1, 100) * np.random.choice([-1, 1], [1])[0]

    def paddle_func(self, x, y):
        if False:
            i = 10
            return i + 15
        return paddle.floor_divide(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            for i in range(10):
                print('nop')
        return builder.floor_divide(x, y, axis)

class TestModOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            i = 10
            return i + 15
        return paddle.mod(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            for i in range(10):
                print('nop')
        return builder.mod(x, y, axis)

class TestModCase1(TestModOp):

    def get_x_data(self):
        if False:
            print('Hello World!')
        return self.random([32, 64], 'int32', 1, 100) * np.random.choice([-1, 1], [1])[0]

    def get_y_data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.random([32, 64], 'int32', 1, 100) * np.random.choice([-1, 1], [1])[0]

class TestRemainderOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            i = 10
            return i + 15
        return paddle.remainder(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            while True:
                i = 10
        return builder.mod(x, y, axis)

class TestRemainderCase1(TestRemainderOp):

    def get_x_data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.random([32, 64], 'int32', 1, 100) * np.random.choice([-1, 1], [1])[0]

    def get_y_data(self):
        if False:
            return 10
        return self.random([32, 64], 'int32', 1, 100) * np.random.choice([-1, 1], [1])[0]

class TestMaxOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        return paddle.maximum(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            for i in range(10):
                print('nop')
        return builder.max(x, y, axis)

class TestMinOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        return paddle.minimum(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            while True:
                i = 10
        return builder.min(x, y, axis)

class TestLogicalAndOp(TestBinaryOp):

    def get_x_data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.random([32, 64], 'bool')

    def get_y_data(self):
        if False:
            print('Hello World!')
        return self.random([32, 64], 'bool')

    def paddle_func(self, x, y):
        if False:
            i = 10
            return i + 15
        return paddle.logical_and(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            for i in range(10):
                print('nop')
        return builder.logical_and(x, y, axis)

class TestLogicalOrOp(TestBinaryOp):

    def get_x_data(self):
        if False:
            i = 10
            return i + 15
        return self.random([32, 64], 'bool')

    def get_y_data(self):
        if False:
            i = 10
            return i + 15
        return self.random([32, 64], 'bool')

    def paddle_func(self, x, y):
        if False:
            i = 10
            return i + 15
        return paddle.logical_or(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            while True:
                i = 10
        return builder.logical_or(x, y, axis)

class TestLogicalXorOp(TestBinaryOp):

    def get_x_data(self):
        if False:
            return 10
        return self.random([32, 64], 'bool')

    def get_y_data(self):
        if False:
            print('Hello World!')
        return self.random([32, 64], 'bool')

    def paddle_func(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        return paddle.logical_xor(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            return 10
        return builder.logical_xor(x, y, axis)

class TestBitwiseAndOp(TestBinaryOp):

    def get_x_data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        if False:
            while True:
                i = 10
        return self.random([32, 64], 'int32', 1, 10000)

    def paddle_func(self, x, y):
        if False:
            i = 10
            return i + 15
        return paddle.bitwise_and(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            i = 10
            return i + 15
        return builder.bitwise_and(x, y, axis)

class TestBitwiseOrOp(TestBinaryOp):

    def get_x_data(self):
        if False:
            return 10
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        if False:
            i = 10
            return i + 15
        return self.random([32, 64], 'int32', 1, 10000)

    def paddle_func(self, x, y):
        if False:
            while True:
                i = 10
        return paddle.bitwise_or(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            for i in range(10):
                print('nop')
        return builder.bitwise_or(x, y, axis)

class TestBitwiseXorOp(TestBinaryOp):

    def get_x_data(self):
        if False:
            i = 10
            return i + 15
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        if False:
            return 10
        return self.random([32, 64], 'int32', 1, 10000)

    def paddle_func(self, x, y):
        if False:
            i = 10
            return i + 15
        return paddle.bitwise_xor(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            i = 10
            return i + 15
        return builder.bitwise_xor(x, y, axis)

class TestEqualOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            i = 10
            return i + 15
        return paddle.equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            while True:
                i = 10
        return builder.equal(x, y, axis)

class TestNotEqualOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        return paddle.not_equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            return 10
        return builder.not_equal(x, y, axis)

class TestGreaterThanOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            print('Hello World!')
        return paddle.greater_than(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            for i in range(10):
                print('nop')
        return builder.greater_than(x, y, axis)

class TestLessThanOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            i = 10
            return i + 15
        return paddle.less_than(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            print('Hello World!')
        return builder.less_than(x, y, axis)

class TestGreaterEqualOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            print('Hello World!')
        return paddle.greater_equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            i = 10
            return i + 15
        return builder.greater_equal(x, y, axis)

class TestLessEqualOp(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            return 10
        return paddle.less_equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            return 10
        return builder.less_equal(x, y, axis)

class TestAtan2Op(TestBinaryOp):

    def paddle_func(self, x, y):
        if False:
            return 10
        return paddle.atan2(x, y)

    def cinn_func(self, builder, x, y, axis):
        if False:
            i = 10
            return i + 15
        return builder.atan2(x, y, axis)
if __name__ == '__main__':
    unittest.main()