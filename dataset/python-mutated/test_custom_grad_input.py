import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg

class TestTensorBackward(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._dtypes = ['float32', 'float64']
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_tensor_backward(self):
        if False:
            i = 10
            return i + 15
        for dtype in self._dtypes:
            x = np.random.random([2, 100]).astype(dtype)
            y = np.random.random([100, 2]).astype(dtype)
            z = np.matmul(x, y)
            grad = np.random.random(z.shape).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = paddle.to_tensor(y)
                    z_tensor = paddle.matmul(x_tensor, y_tensor)
                    grad_tensor = paddle.to_tensor(grad)
                    z_tensor.backward(grad_tensor)
                    x_grad = np.matmul(grad, y.T)
                    np.testing.assert_allclose(x_grad, x_tensor.grad.numpy(), rtol=1e-05)

class TestBackwardAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._dtypes = ['float32', 'float64']
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_backward_api(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self._dtypes:
            x = np.random.random([2, 2]).astype(dtype)
            y = np.random.random([2, 2]).astype(dtype)
            z = np.matmul(x, y)
            grad = np.random.random(z.shape).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = paddle.to_tensor(y)
                    z_tensor1 = paddle.matmul(x_tensor, y_tensor)
                    z_tensor2 = paddle.matmul(x_tensor, y_tensor)
                    grad_tensor = paddle.to_tensor(grad)
                    paddle.autograd.backward([z_tensor1, z_tensor2], [grad_tensor, grad_tensor], True)
                    x_grad = np.matmul(grad, y.T)
                    np.testing.assert_allclose(x_grad * 2, x_tensor.grad.numpy(), rtol=1e-05)

    def test_backward_single_tensor(self):
        if False:
            i = 10
            return i + 15
        for dtype in self._dtypes:
            x = np.random.random([2, 2]).astype(dtype)
            y = np.random.random([2, 2]).astype(dtype)
            z = np.matmul(x, y)
            grad = np.random.random(z.shape).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = paddle.to_tensor(y)
                    z_tensor1 = paddle.matmul(x_tensor, y_tensor)
                    grad_tensor = paddle.to_tensor(grad)
                    paddle.autograd.backward(z_tensor1, grad_tensor, True)
                    x_grad = np.matmul(grad, y.T)
                    np.testing.assert_allclose(x_grad, x_tensor.grad.numpy(), rtol=1e-05)

    def test_backward_none_grad_tensor(self):
        if False:
            return 10
        for dtype in self._dtypes:
            x = np.random.random([2, 2]).astype(dtype)
            y = np.random.random([2, 2]).astype(dtype)
            z = np.matmul(x, y)
            grad = np.ones(z.shape).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = paddle.to_tensor(y)
                    z_tensor1 = paddle.matmul(x_tensor, y_tensor)
                    paddle.autograd.backward(z_tensor1, None)
                    x_grad = np.matmul(grad, y.T)
                    np.testing.assert_allclose(x_grad, x_tensor.grad.numpy(), rtol=1e-05)

    def test_backward_accumulator_with_init_grad(self):
        if False:
            print('Hello World!')
        for dtype in self._dtypes:
            x = np.random.random([10]).astype(dtype)
            y_grad = np.random.random([10]).astype(dtype)
            z_grad = np.random.random([10]).astype(dtype)
            self._places = [paddle.CPUPlace()]
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = x_tensor ** 2
                    z_tensor = y_tensor ** 3
                    y_grad_tensor = paddle.to_tensor(y_grad)
                    z_grad_tensor = paddle.to_tensor(z_grad)
                    paddle.autograd.backward([y_tensor, z_tensor], [y_grad_tensor, z_grad_tensor])
                    y = x ** 2
                    z = x ** 3
                    x_grad = 2 * x * (y_grad + 3 * y * y * z_grad)
                    np.testing.assert_allclose(x_grad, x_tensor.grad.numpy(), rtol=1e-05)
if __name__ == '__main__':
    unittest.main()