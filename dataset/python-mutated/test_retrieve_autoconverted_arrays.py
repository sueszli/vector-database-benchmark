import numpy as np
from numba import cuda
from numba.cuda.args import wrap_arg
from numba.cuda.testing import CUDATestCase
import unittest

class DefaultIn(object):

    def prepare_args(self, ty, val, **kwargs):
        if False:
            while True:
                i = 10
        return (ty, wrap_arg(val, default=cuda.In))

def nocopy(kernel):
    if False:
        i = 10
        return i + 15
    kernel.extensions.append(DefaultIn())
    return kernel

def set_array_to_three(arr):
    if False:
        i = 10
        return i + 15
    arr[0] = 3

def set_record_to_three(rec):
    if False:
        print('Hello World!')
    rec[0]['b'] = 3
recordtype = np.dtype([('b', np.int32)], align=True)

class TestRetrieveAutoconvertedArrays(CUDATestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.set_array_to_three = cuda.jit(set_array_to_three)
        self.set_array_to_three_nocopy = nocopy(cuda.jit(set_array_to_three))
        self.set_record_to_three = cuda.jit(set_record_to_three)
        self.set_record_to_three_nocopy = nocopy(cuda.jit(set_record_to_three))

    def test_array_inout(self):
        if False:
            print('Hello World!')
        host_arr = np.zeros(1, dtype=np.int64)
        self.set_array_to_three[1, 1](cuda.InOut(host_arr))
        self.assertEqual(3, host_arr[0])

    def test_array_in(self):
        if False:
            i = 10
            return i + 15
        host_arr = np.zeros(1, dtype=np.int64)
        self.set_array_to_three[1, 1](cuda.In(host_arr))
        self.assertEqual(0, host_arr[0])

    def test_array_in_from_config(self):
        if False:
            i = 10
            return i + 15
        host_arr = np.zeros(1, dtype=np.int64)
        self.set_array_to_three_nocopy[1, 1](host_arr)
        self.assertEqual(0, host_arr[0])

    def test_array_default(self):
        if False:
            while True:
                i = 10
        host_arr = np.zeros(1, dtype=np.int64)
        self.set_array_to_three[1, 1](host_arr)
        self.assertEqual(3, host_arr[0])

    def test_record_in(self):
        if False:
            i = 10
            return i + 15
        host_rec = np.zeros(1, dtype=recordtype)
        self.set_record_to_three[1, 1](cuda.In(host_rec))
        self.assertEqual(0, host_rec[0]['b'])

    def test_record_inout(self):
        if False:
            for i in range(10):
                print('nop')
        host_rec = np.zeros(1, dtype=recordtype)
        self.set_record_to_three[1, 1](cuda.InOut(host_rec))
        self.assertEqual(3, host_rec[0]['b'])

    def test_record_default(self):
        if False:
            for i in range(10):
                print('nop')
        host_rec = np.zeros(1, dtype=recordtype)
        self.set_record_to_three[1, 1](host_rec)
        self.assertEqual(3, host_rec[0]['b'])

    def test_record_in_from_config(self):
        if False:
            return 10
        host_rec = np.zeros(1, dtype=recordtype)
        self.set_record_to_three_nocopy[1, 1](host_rec)
        self.assertEqual(0, host_rec[0]['b'])
if __name__ == '__main__':
    unittest.main()