import unittest
from op_mapper_test import OpMapperTest
import paddle

class TestPowOp(OpMapperTest):

    def init_input_data(self):
        if False:
            return 10
        self.feed_data = {'x': self.random([32, 64], 'float32'), 'factor': self.random([1], 'float32', 0.0, 4.0)}

    def set_op_type(self):
        if False:
            return 10
        return 'pow'

    def set_op_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name='x', shape=self.feed_data['x'].shape, dtype=self.feed_data['x'].dtype)
        factor = paddle.static.data(name='factor', shape=self.feed_data['factor'].shape, dtype=self.feed_data['factor'].dtype)
        return {'X': [x], 'FactorTensor': [factor]}

    def set_op_attrs(self):
        if False:
            return 10
        return {}

    def set_op_outputs(self):
        if False:
            i = 10
            return i + 15
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_outputs_and_grads()

class TestPowCase1(TestPowOp):

    def init_input_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_data = {'x': self.random([32, 64], 'int32', 2, 10), 'factor': self.random([1], 'int32', 0, 5)}

class TestPowCase2(TestPowOp):

    def init_input_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_data = {'x': self.random([32, 64], 'int32', 2, 10), 'factor': self.random([1], 'int32', 0, 5)}

class TestPowOpInFactorAttr(TestPowOp):

    def set_op_inputs(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name='x', shape=self.feed_data['x'].shape, dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        return {'factor': float(2)}
if __name__ == '__main__':
    unittest.main()