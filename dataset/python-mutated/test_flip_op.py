import unittest
from op_mapper_test import OpMapperTest
import paddle

class TestFlipOp(OpMapperTest):

    def init_input_data(self):
        if False:
            print('Hello World!')
        self.feed_data = {'x': self.random([3, 2, 4], 'float32')}

    def set_op_type(self):
        if False:
            i = 10
            return i + 15
        return 'flip'

    def set_op_inputs(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name='x', shape=self.feed_data['x'].shape, dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        return {'axis': [0, 1]}

    def set_op_outputs(self):
        if False:
            print('Hello World!')
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        if False:
            print('Hello World!')
        self.check_outputs_and_grads()

class TestFlipOpAxis(TestFlipOp):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        return {'axis': [0, 2]}
if __name__ == '__main__':
    unittest.main()