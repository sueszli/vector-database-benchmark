"""This is unit test of Test shuffle_batch Op."""
import os
import unittest
import numpy as np
from op_test import OpTest
from paddle import base

class TestShuffleBatchOpBase(OpTest):

    def gen_random_array(self, shape, low=0, high=1):
        if False:
            return 10
        rnd = (high - low) * np.random.random(shape) + low
        return rnd.astype(self.dtype)

    def get_shape(self):
        if False:
            return 10
        return (10, 10, 5)

    def _get_places(self):
        if False:
            while True:
                i = 10
        if os.name == 'nt':
            return [base.CPUPlace()]
        return super()._get_places()

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'shuffle_batch'
        self.dtype = np.float64
        self.shape = self.get_shape()
        x = self.gen_random_array(self.shape)
        seed = np.random.random_integers(low=10, high=100, size=(1,)).astype('int64')
        self.inputs = {'X': x, 'Seed': seed}
        self.outputs = {'Out': np.array([]).astype(x.dtype), 'ShuffleIdx': np.array([]).astype('int64'), 'SeedOut': np.array([]).astype(seed.dtype)}
        self.attrs = {'startup_seed': 1}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        if False:
            return 10
        x = np.copy(self.inputs['X'])
        y = None
        for out in outs:
            if out.shape == x.shape:
                y = np.copy(out)
                break
        assert y is not None
        sort_x = self.sort_array(x)
        sort_y = self.sort_array(y)
        np.testing.assert_array_equal(sort_x, sort_y)

    def sort_array(self, array):
        if False:
            return 10
        shape = array.shape
        new_shape = [-1, shape[-1]]
        arr_list = np.reshape(array, new_shape).tolist()
        arr_list.sort(key=lambda x: x[0])
        return np.reshape(np.array(arr_list), shape)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestShuffleBatchOp2(TestShuffleBatchOpBase):

    def get_shape(self):
        if False:
            print('Hello World!')
        return (4, 30)
if __name__ == '__main__':
    unittest.main()