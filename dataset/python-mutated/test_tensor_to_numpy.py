import unittest
import numpy as np
from paddle import base

class TensorToNumpyTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = [11, 25, 32, 43]

    def test_main(self):
        if False:
            while True:
                i = 10
        dtypes = ['float32', 'float64', 'int32', 'int64', 'uint8', 'int8', 'bool']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
            places.append(base.CUDAPinnedPlace())
        for p in places:
            for dtype in dtypes:
                np_arr = np.reshape(np.array(range(np.prod(self.shape))).astype(dtype), self.shape)
                t = base.LoDTensor()
                t.set(np_arr, p)
                ret_np_arr = np.array(t)
                self.assertEqual(np_arr.shape, ret_np_arr.shape)
                self.assertEqual(np_arr.dtype, ret_np_arr.dtype)
                all_equal = np.all(np_arr == ret_np_arr)
                self.assertTrue(all_equal)
if __name__ == '__main__':
    unittest.main()