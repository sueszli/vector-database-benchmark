import unittest
import numpy as np
from op_test import OpTest

def PolygonBoxRestore(input):
    if False:
        print('Hello World!')
    shape = input.shape
    batch_size = shape[0]
    geo_channels = shape[1]
    h = shape[2]
    w = shape[3]
    h_indexes = np.array(list(range(h)) * w).reshape([w, h]).transpose()[np.newaxis, :]
    w_indexes = np.array(list(range(w)) * h).reshape([h, w])[np.newaxis, :]
    indexes = np.concatenate((w_indexes, h_indexes))[np.newaxis, :]
    indexes = indexes.repeat([geo_channels / 2], axis=0)[np.newaxis, :]
    indexes = indexes.repeat([batch_size], axis=0)
    return indexes.reshape(input.shape) * 4 - input

class TestPolygonBoxRestoreOp(OpTest):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.input_shape = (1, 8, 2, 2)

    def setUp(self):
        if False:
            return 10
        self.config()
        self.op_type = 'polygon_box_transform'
        input = np.random.random(self.input_shape).astype('float32')
        self.inputs = {'Input': input}
        output = PolygonBoxRestore(input)
        self.outputs = {'Output': output}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()

class TestCase1(TestPolygonBoxRestoreOp):

    def config(self):
        if False:
            return 10
        self.input_shape = (2, 10, 3, 2)

class TestCase2(TestPolygonBoxRestoreOp):

    def config(self):
        if False:
            return 10
        self.input_shape = (3, 12, 4, 5)
if __name__ == '__main__':
    unittest.main()