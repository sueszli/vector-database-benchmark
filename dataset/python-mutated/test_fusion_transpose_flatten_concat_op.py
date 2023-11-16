import unittest
import numpy as np
from op_test import OpTest
from paddle.base import core

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFusionTransposeFlattenConcationOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_test_case()
        self.op_type = 'fusion_transpose_flatten_concat'
        ins = []
        flats = []
        for i in range(len(self.shapes)):
            in_shape = self.shapes[i]
            a = np.random.random(in_shape).astype('float32')
            ins.append(('x%d' % i, a))
            b = a.transpose(self.trans_axis)
            flat_shape = (np.prod(b.shape[:self.flatten_axis]), np.prod(b.shape[self.flatten_axis:]))
            c = b.reshape(flat_shape)
            flats.append(c)
        out = np.concatenate(flats, axis=self.concat_axis)
        self.inputs = {'X': ins}
        self.attrs = {'trans_axis': list(self.trans_axis), 'flatten_axis': self.flatten_axis, 'concat_axis': self.concat_axis}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, 1e-06, check_dygraph=False)

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.shapes = [(3, 4, 17, 17), (3, 8, 7, 7), (3, 12, 5, 5)]
        self.trans_axis = (0, 2, 3, 1)
        self.flatten_axis = 1
        self.concat_axis = 1

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestCase1(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.shapes = [(3, 4, 18, 17), (3, 8, 18, 7), (6, 12, 9, 5)]
        self.trans_axis = (0, 2, 3, 1)
        self.flatten_axis = 2
        self.concat_axis = 1

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestCase2(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.shapes = [(3, 8, 20, 17), (3, 8, 19, 17), (3, 8, 40, 17)]
        self.trans_axis = (0, 2, 3, 1)
        self.flatten_axis = 2
        self.concat_axis = 0

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestCase3(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.shapes = [(3, 8, 20, 17), (3, 8, 19, 17), (3, 8, 40, 17)]
        self.trans_axis = (0, 3, 2, 1)
        self.flatten_axis = 1
        self.concat_axis = 1

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestCase4(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        if False:
            return 10
        self.shapes = [(3, 8, 9, 17), (8, 3, 9, 17), (4, 6, 9, 17)]
        self.trans_axis = (0, 2, 1, 3)
        self.flatten_axis = 3
        self.concat_axis = 1

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestCase5(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.shapes = [(3, 8, 9, 17, 2), (3, 8, 2, 17, 9), (3, 17, 9, 8, 2)]
        self.trans_axis = (0, 2, 1, 4, 3)
        self.flatten_axis = 1
        self.concat_axis = 1
if __name__ == '__main__':
    unittest.main()