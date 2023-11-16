import unittest
import numpy as np
import paddle
from paddle import base

class TensorToListTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.shape = [11, 25, 32, 43]

    def test_tensor_tolist(self):
        if False:
            for i in range(10):
                print('nop')
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
            places.append(base.CUDAPinnedPlace())
        for p in places:
            np_arr = np.reshape(np.array(range(np.prod(self.shape))), self.shape)
            expectlist = np_arr.tolist()
            t = paddle.to_tensor(np_arr, place=p)
            tensorlist = t.tolist()
            self.assertEqual(tensorlist, expectlist)
if __name__ == '__main__':
    unittest.main()