import unittest
from op_mapper_test import OpMapperTest
import paddle

class TestOneHotV2Op(OpMapperTest):

    def init_input_data(self):
        if False:
            return 10
        self.feed_data = {'x': self.random([1, 32], 'int32')}
        self.depth = 10
        self.dtype = 'float32'
        self.allow_out_of_range = False

    def set_op_type(self):
        if False:
            print('Hello World!')
        return 'one_hot_v2'

    def set_op_inputs(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name='x', shape=self.feed_data['x'].shape, dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        if False:
            return 10
        return {'depth': self.depth, 'dtype': self.nptype2paddledtype(self.dtype), 'allow_out_of_range': self.allow_out_of_range}

    def set_op_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        if False:
            return 10
        self.check_outputs_and_grads(all_equal=True)

class TestOneHotV2OpCase1(TestOneHotV2Op):

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.feed_data = {'x': self.random([32, 64], 'int32')}
        self.depth = 64
        self.dtype = 'int32'
        self.allow_out_of_range = False

class TestOneHotV2OpCase2(TestOneHotV2Op):

    def init_input_data(self):
        if False:
            return 10
        self.feed_data = {'x': self.random([32, 64, 1], 'int64')}
        self.depth = 1
        self.dtype = 'int64'
        self.allow_out_of_range = True
if __name__ == '__main__':
    unittest.main()