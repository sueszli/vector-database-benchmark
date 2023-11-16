import unittest
import numpy as np
import paddle
from paddle import base

class TensorFillDiagonal_Test(unittest.TestCase):

    def test_dim2_normal(self):
        if False:
            return 10
        expected_np = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1]]).astype('float32')
        expected_grad = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).astype('float32')
        typelist = ['float32', 'float64', 'int32', 'int64']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((3, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                y.fill_diagonal_(1, offset=0, wrap=True)
                loss = y.sum()
                loss.backward()
                self.assertEqual((y.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual((y.grad.numpy().astype('float32') == expected_grad).all(), True)

    def test_offset(self):
        if False:
            return 10
        expected_np = np.array([[2, 2, 1], [2, 2, 2], [2, 2, 2]]).astype('float32')
        expected_grad = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]]).astype('float32')
        typelist = ['float32', 'float64', 'int32', 'int64']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((3, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                y.fill_diagonal_(1, offset=2, wrap=True)
                loss = y.sum()
                loss.backward()
                self.assertEqual((y.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual((y.grad.numpy().astype('float32') == expected_grad).all(), True)

    def test_bool(self):
        if False:
            for i in range(10):
                print('nop')
        expected_np = np.array([[False, True, True], [True, False, True], [True, True, False]])
        typelist = ['bool']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((3, 3), dtype=dtype)
                x.stop_gradient = True
                x.fill_diagonal_(0, offset=0, wrap=True)
                self.assertEqual((x.numpy() == expected_np).all(), True)

    def test_dim2_unnormal_wrap(self):
        if False:
            i = 10
            return i + 15
        expected_np = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 2], [1, 2, 2], [2, 1, 2], [2, 2, 1]]).astype('float32')
        expected_grad = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]).astype('float32')
        typelist = ['float32', 'float64', 'int32', 'int64']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((7, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                y.fill_diagonal_(1, offset=0, wrap=True)
                loss = y.sum()
                loss.backward()
                self.assertEqual((y.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual((y.grad.numpy().astype('float32') == expected_grad).all(), True)

    def test_dim2_unnormal_unwrap(self):
        if False:
            while True:
                i = 10
        expected_np = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]).astype('float32')
        expected_grad = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype('float32')
        typelist = ['float32', 'float64', 'int32', 'int64']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((7, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                y.fill_diagonal_(1, offset=0, wrap=False)
                loss = y.sum()
                loss.backward()
                self.assertEqual((y.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual((y.grad.numpy().astype('float32') == expected_grad).all(), True)

    def test_dim_larger2_normal(self):
        if False:
            return 10
        expected_np = np.array([[[1, 2, 2], [2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 1, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2], [2, 2, 1]]]).astype('float32')
        expected_grad = np.array([[[0, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 0]]]).astype('float32')
        typelist = ['float32', 'float64', 'int32', 'int64']
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for (idx, p) in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((3, 3, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                y.fill_diagonal_(1, offset=0, wrap=True)
                loss = y.sum()
                loss.backward()
                self.assertEqual((y.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual((y.grad.numpy().astype('float32') == expected_grad).all(), True)
if __name__ == '__main__':
    unittest.main()