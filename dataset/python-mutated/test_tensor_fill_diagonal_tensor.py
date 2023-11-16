import unittest
import numpy as np
import paddle
from paddle import base

class TensorFillDiagTensor_Test(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.typelist = ['float32', 'float64', 'int32', 'int64']
        self.places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_dim2(self):
        if False:
            print('Hello World!')
        expected_np = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 2]]).astype('float32')
        expected_grad = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).astype('float32')
        for (idx, p) in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.ones((3,), dtype=dtype)
                var = np.random.random() + 1
                x = paddle.ones((4, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                ny = y.fill_diagonal_tensor(v, offset=0, dim1=0, dim2=1)
                loss = ny.sum()
                loss.backward()
                self.assertEqual((ny.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual((y.grad.numpy().astype('float32') == expected_grad).all(), True)

    def test_dim2_offset_1(self):
        if False:
            return 10
        expected_np = np.array([[2, 2, 2], [1, 2, 2], [2, 1, 2], [2, 2, 1]]).astype('float32')
        expected_grad = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]).astype('float32')
        for (idx, p) in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.ones((3,), dtype=dtype)
                var = np.random.random() + 1
                x = paddle.ones((4, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                ny = y.fill_diagonal_tensor(v, offset=-1, dim1=0, dim2=1)
                loss = ny.sum()
                loss.backward()
                self.assertEqual((ny.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual((y.grad.numpy().astype('float32') == expected_grad).all(), True)

    def test_dim2_offset1(self):
        if False:
            while True:
                i = 10
        expected_np = np.array([[2, 1, 2], [2, 2, 1], [2, 2, 2], [2, 2, 2]]).astype('float32')
        expected_grad = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1]]).astype('float32')
        for (idx, p) in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.ones((2,), dtype=dtype)
                var = np.random.random() + 1
                x = paddle.ones((4, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                ny = y.fill_diagonal_tensor(v, offset=1, dim1=0, dim2=1)
                loss = ny.sum()
                loss.backward()
                self.assertEqual((ny.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual((y.grad.numpy().astype('float32') == expected_grad).all(), True)

    def test_dim4(self):
        if False:
            i = 10
            return i + 15
        expected_np = np.array([[[[0, 3], [2, 2], [2, 2]], [[2, 2], [1, 4], [2, 2]], [[2, 2], [2, 2], [2, 5]], [[2, 2], [2, 2], [2, 2]]], [[[6, 9], [2, 2], [2, 2]], [[2, 2], [7, 10], [2, 2]], [[2, 2], [2, 2], [8, 11]], [[2, 2], [2, 2], [2, 2]]]]).astype('float32')
        expected_grad = np.array([[[[0, 0], [1, 1], [1, 1]], [[1, 1], [0, 0], [1, 1]], [[1, 1], [1, 1], [0, 0]], [[1, 1], [1, 1], [1, 1]]], [[[0, 0], [1, 1], [1, 1]], [[1, 1], [0, 0], [1, 1]], [[1, 1], [1, 1], [0, 0]], [[1, 1], [1, 1], [1, 1]]]]).astype('float32')
        for (idx, p) in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.to_tensor(np.arange(12).reshape(2, 2, 3), dtype=dtype)
                var = np.random.random() + 1
                x = paddle.ones((2, 4, 3, 2), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                ny = y.fill_diagonal_tensor(v, offset=0, dim1=1, dim2=2)
                loss = ny.sum()
                loss.backward()
                self.assertEqual((ny.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual((y.grad.numpy().astype('float32') == expected_grad).all(), True)

    def test_largedim(self):
        if False:
            print('Hello World!')
        if len(self.places) > 1:
            bsdim = 1024
            fsdim = 128
            paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.arange(bsdim * fsdim, dtype=dtype).reshape((bsdim, fsdim))
                y = paddle.ones((bsdim, fsdim, fsdim), dtype=dtype)
                y.stop_gradient = False
                y = y * 2
                y.retain_grads()
                ny = y.fill_diagonal_tensor(v, offset=0, dim1=1, dim2=2)
                loss = ny.sum()
                loss.backward()
                expected_pred = v - 2
                expected_pred = paddle.diag_embed(expected_pred) + 2
                expected_grad = paddle.ones(v.shape, dtype=dtype) - 2
                expected_grad = paddle.diag_embed(expected_grad) + 1
                self.assertEqual((ny == expected_pred).all(), True)
                self.assertEqual((y.grad == expected_grad).all(), True)
if __name__ == '__main__':
    unittest.main()