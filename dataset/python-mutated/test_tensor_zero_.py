import unittest
import numpy as np
import paddle
from paddle import base

class TensorFill_Test(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.shape = [32, 32]

    def test_tensor_fill_true(self):
        if False:
            for i in range(10):
                print('nop')
        typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
            places.append(base.CUDAPinnedPlace())
        for p in places:
            np_arr = np.reshape(np.array(range(np.prod(self.shape))), self.shape)
            for dtype in typelist:
                tensor = paddle.to_tensor(np_arr, place=p, dtype=dtype)
                target = tensor.numpy()
                target[...] = 0
                tensor.zero_()
                self.assertEqual((tensor.numpy() == target).all().item(), True)
if __name__ == '__main__':
    unittest.main()