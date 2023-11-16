import unittest
from op_mapper_test import OpMapperTest
import paddle

class TestSqueezeOp(OpMapperTest):

    def init_input_data(self):
        if False:
            i = 10
            return i + 15
        self.feed_data = {'x': self.random([5, 1, 10], 'float32')}

    def set_op_type(self):
        if False:
            while True:
                i = 10
        return 'squeeze2'

    def set_op_inputs(self):
        if False:
            return 10
        x = paddle.static.data(name='x', shape=self.feed_data['x'].shape, dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        return {'axes': [1]}

    def set_op_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        return {'Out': [str(self.feed_data['x'].dtype)], 'XShape': [str(self.feed_data['x'].dtype)]}

    def skip_check_outputs(self):
        if False:
            print('Hello World!')
        return {'XShape'}

    def test_check_results(self):
        if False:
            i = 10
            return i + 15
        self.check_outputs_and_grads()

class TestSqueezeAxesEmpty(TestSqueezeOp):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        return {'axes': []}
if __name__ == '__main__':
    unittest.main()