import unittest
from op_mapper_test import OpMapperTest
import paddle

class TestArgmaxOp(OpMapperTest):

    def init_input_data(self):
        if False:
            return 10
        self.axis = 1
        self.shape = [2, 3, 4]
        self.input_dtype = 'float32'
        self.output_dtype = 'int64'
        self.flatten = False
        self.keepdims = False
        self.feed_data = {'x': self.random(self.shape, self.input_dtype)}

    def set_op_type(self):
        if False:
            while True:
                i = 10
        return 'arg_max'

    def set_op_inputs(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name='x', shape=self.feed_data['x'].shape, dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        return {'axis': self.axis, 'flatten': self.flatten, 'keepdims': self.keepdims, 'dtype': self.nptype2paddledtype(self.output_dtype)}

    def set_op_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        return {'Out': [str(self.output_dtype)]}

    def test_check_results(self):
        if False:
            print('Hello World!')
        self.check_outputs_and_grads(all_equal=True)

class TestArgmaxCase1(TestArgmaxOp):
    """
    Test case with negative axis and True flatten and int64 output dtype
    """

    def init_input_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.axis = -1
        self.shape = [2, 3, 4]
        self.input_dtype = 'float32'
        self.output_dtype = 'int64'
        self.keepdims = False
        self.flatten = True
        self.feed_data = {'x': self.random(self.shape, self.input_dtype)}

class TestArgmaxCase2(TestArgmaxOp):
    """
    Test case with true keepdims
    """

    def init_input_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.axis = -1
        self.shape = [2, 3, 4]
        self.input_dtype = 'float32'
        self.output_dtype = 'int32'
        self.flatten = False
        self.keepdims = True
        self.feed_data = {'x': self.random(self.shape, self.input_dtype)}

class TestArgmaxCase3(TestArgmaxOp):
    """
    Test case with input_dtype float64
    """

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.axis = -1
        self.shape = [2, 3, 4]
        self.input_dtype = 'float64'
        self.output_dtype = 'int32'
        self.flatten = False
        self.keepdims = True
        self.feed_data = {'x': self.random(self.shape, self.input_dtype)}
if __name__ == '__main__':
    unittest.main()