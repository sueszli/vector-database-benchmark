import unittest
import numpy as np
import paddle
from paddle import base

class TensorFill_Test(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = [32, 32]

    def test_tensor_fill_true(self):
        if False:
            while True:
                i = 10
        typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
            places.append(base.CUDAPinnedPlace())
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            np_arr = np.reshape(np.array(range(np.prod(self.shape))), self.shape)
            for dtype in typelist:
                var = 1.0
                tensor = paddle.to_tensor(np_arr, place=p, dtype=dtype)
                target = tensor.numpy()
                target[...] = var
                tensor.fill_(var)
                self.assertEqual((tensor.numpy() == target).all(), True)

    def test_tensor_fill_backward(self):
        if False:
            for i in range(10):
                print('nop')
        typelist = ['float32']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
            places.append(base.CUDAPinnedPlace())
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            np_arr = np.reshape(np.array(range(np.prod(self.shape))), self.shape)
            for dtype in typelist:
                var = 1
                tensor = paddle.to_tensor(np_arr, place=p, dtype=dtype)
                tensor.stop_gradient = False
                y = tensor * 2
                y.retain_grads()
                y.fill_(var)
                loss = y.sum()
                loss.backward()
                self.assertEqual((y.grad.numpy() == 0).all().item(), True)

    def test_errors(self):
        if False:
            i = 10
            return i + 15

        def test_list():
            if False:
                i = 10
                return i + 15
            x = paddle.to_tensor([2, 3, 4])
            x.fill_([1])
        self.assertRaises(TypeError, test_list)
if __name__ == '__main__':
    unittest.main()