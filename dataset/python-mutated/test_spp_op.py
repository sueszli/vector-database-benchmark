import unittest
import numpy as np
from op_test import OpTest
from test_pool2d_op import avg_pool2D_forward_naive, max_pool2D_forward_naive

class TestSppOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'spp'
        self.init_test_case()
        (nsize, csize, hsize, wsize) = self.shape
        data = np.array(list(range(nsize * csize * hsize * wsize)))
        input = data.reshape(self.shape)
        input_random = np.random.random(self.shape).astype('float64')
        input = input + input_random
        out_level_flatten = []
        for i in range(self.pyramid_height):
            bins = np.power(2, i)
            kernel_size = [0, 0]
            padding = [0, 0]
            kernel_size[0] = np.ceil(hsize / bins.astype('double')).astype('int32')
            padding[0] = ((kernel_size[0] * bins - hsize + 1) / 2).astype('int32')
            kernel_size[1] = np.ceil(wsize / bins.astype('double')).astype('int32')
            padding[1] = ((kernel_size[1] * bins - wsize + 1) / 2).astype('int32')
            out_level = self.pool2D_forward_naive(input, kernel_size, kernel_size, padding)
            out_level_flatten.append(out_level.reshape(nsize, bins * bins * csize))
            if i == 0:
                output = out_level_flatten[i]
            else:
                output = np.concatenate((output, out_level_flatten[i]), 1)
        self.inputs = {'X': input.astype('float64')}
        self.attrs = {'pyramid_height': self.pyramid_height, 'pooling_type': self.pool_type}
        self.outputs = {'Out': output.astype('float64')}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [3, 2, 16, 16]
        self.pyramid_height = 3
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.pool_type = 'max'

class TestCase2(TestSppOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [3, 2, 16, 16]
        self.pyramid_height = 3
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.pool_type = 'avg'
if __name__ == '__main__':
    unittest.main()