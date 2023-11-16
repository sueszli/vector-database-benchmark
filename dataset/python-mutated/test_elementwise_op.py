import unittest
from op_mapper_test import OpMapperTest
import paddle

class TestElementwiseOp(OpMapperTest):

    def init_input_data(self):
        if False:
            print('Hello World!')
        self.feed_data = {'x': self.random([32, 64], 'float32'), 'y': self.random([32, 64], 'float32')}

    def set_op_type(self):
        if False:
            return 10
        return 'elementwise_add'

    def set_op_inputs(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name='x', shape=self.feed_data['x'].shape, dtype=self.feed_data['x'].dtype)
        y = paddle.static.data(name='y', shape=self.feed_data['y'].shape, dtype=self.feed_data['y'].dtype)
        return {'X': [x], 'Y': [y]}

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        return {'axis': -1}

    def set_op_outputs(self):
        if False:
            print('Hello World!')
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        if False:
            print('Hello World!')
        self.check_outputs_and_grads()

class TestAddOp(TestElementwiseOp):

    def set_op_type(self):
        if False:
            print('Hello World!')
        return 'elementwise_add'

class TestSubOp(TestElementwiseOp):

    def set_op_type(self):
        if False:
            for i in range(10):
                print('nop')
        return 'elementwise_sub'

class TestDivOp(TestElementwiseOp):

    def set_op_type(self):
        if False:
            while True:
                i = 10
        return 'elementwise_div'

class TestMulOp(TestElementwiseOp):

    def set_op_type(self):
        if False:
            for i in range(10):
                print('nop')
        return 'elementwise_mul'

class TestPowOp(TestElementwiseOp):

    def set_op_type(self):
        if False:
            i = 10
            return i + 15
        return 'elementwise_pow'

class TestModOp(TestElementwiseOp):

    def set_op_type(self):
        if False:
            for i in range(10):
                print('nop')
        return 'elementwise_mod'

class TestMaxOp(TestElementwiseOp):

    def set_op_type(self):
        if False:
            for i in range(10):
                print('nop')
        return 'elementwise_max'

class TestMinOp(TestElementwiseOp):

    def set_op_type(self):
        if False:
            i = 10
            return i + 15
        return 'elementwise_min'

class TestFloorDivOpCase1(TestElementwiseOp):

    def init_input_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_data = {'x': self.random([32, 64], low=1, high=10, dtype='int32'), 'y': self.random([32, 64], low=1, high=10, dtype='int32')}

    def set_op_type(self):
        if False:
            return 10
        return 'elementwise_floordiv'

class TestFloorDivOpCase2(TestElementwiseOp):

    def init_input_data(self):
        if False:
            return 10
        self.feed_data = {'x': self.random([32], low=1, high=10, dtype='int64'), 'y': self.random([32], low=1, high=10, dtype='int64')}

    def set_op_type(self):
        if False:
            print('Hello World!')
        return 'elementwise_floordiv'
if __name__ == '__main__':
    unittest.main()