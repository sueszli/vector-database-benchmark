import unittest
from op_mapper_test import OpMapperTest
import paddle

class TestTranspose2Op(OpMapperTest):

    def init_input_dtype(self):
        if False:
            return 10
        self.dtype = 'float32'

    def init_input_data(self):
        if False:
            print('Hello World!')
        self.init_input_dtype()
        self.feed_data = {'x': self.random([2, 3, 4], self.dtype, 0.0, 100.0)}

    def set_op_type(self):
        if False:
            for i in range(10):
                print('nop')
        return 'transpose2'

    def set_op_inputs(self):
        if False:
            return 10
        x = paddle.static.data(name='x', shape=self.feed_data['x'].shape, dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        return {'axis': [0, 2, 1]}

    def set_op_outputs(self):
        if False:
            print('Hello World!')
        return {'Out': [str(self.feed_data['x'].dtype)], 'XShape': [str(self.feed_data['x'].dtype)]}

    def skip_check_outputs(self):
        if False:
            i = 10
            return i + 15
        return {'XShape'}

    def test_check_results(self):
        if False:
            i = 10
            return i + 15
        self.check_outputs_and_grads(all_equal=True)

class TestTranspose2OpInt32(TestTranspose2Op):

    def init_input_dtype(self):
        if False:
            return 10
        self.dtype = 'int32'

class TestTranspose2OpInt64(TestTranspose2Op):

    def init_input_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = 'int64'
if __name__ == '__main__':
    unittest.main()