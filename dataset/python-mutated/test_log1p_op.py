import unittest
from op_mapper_test import OpMapperTest
import paddle

class TestLog1pOp(OpMapperTest):

    def init_input_data(self):
        if False:
            print('Hello World!')
        self.feed_data = {'x': self.random([10, 12, 128, 128], 'float32')}

    def set_op_type(self):
        if False:
            return 10
        return 'log1p'

    def set_op_inputs(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name='x', shape=self.feed_data['x'].shape, dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

    def set_op_outputs(self):
        if False:
            print('Hello World!')
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        if False:
            i = 10
            return i + 15
        self.check_outputs_and_grads()
if __name__ == '__main__':
    unittest.main()