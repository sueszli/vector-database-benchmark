import copy
import unittest
import numpy as np
import paddle

class TestMaxPool3DFunc(unittest.TestCase):

    def setInput(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(0)
        self.dense_x = paddle.randn((1, 4, 4, 4, 4))

    def setKernelSize(self):
        if False:
            print('Hello World!')
        self.kernel_sizes = [3, 3, 3]

    def setStride(self):
        if False:
            print('Hello World!')
        self.strides = [1, 1, 1]

    def setPadding(self):
        if False:
            while True:
                i = 10
        self.paddings = [0, 0, 0]

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setInput()
        self.setKernelSize()
        self.setStride()
        self.setPadding()

    def test(self):
        if False:
            i = 10
            return i + 15
        self.setUp()
        self.dense_x.stop_gradient = False
        sparse_x = self.dense_x.to_sparse_coo(4)
        sparse_out = paddle.sparse.nn.functional.max_pool3d(sparse_x, self.kernel_sizes, stride=self.strides, padding=self.paddings)
        out = sparse_out.to_dense()
        out.backward(out)
        dense_x = copy.deepcopy(self.dense_x)
        dense_out = paddle.nn.functional.max_pool3d(dense_x, self.kernel_sizes, stride=self.strides, padding=self.paddings, data_format='NDHWC')
        dense_out.backward(dense_out)
        np.testing.assert_allclose(dense_out.numpy(), out.numpy())
        np.testing.assert_allclose(dense_x.grad.numpy(), self.dense_x.grad.numpy())

class TestStride(TestMaxPool3DFunc):

    def setStride(self):
        if False:
            for i in range(10):
                print('nop')
        self.strides = 1

class TestPadding(TestMaxPool3DFunc):

    def setPadding(self):
        if False:
            i = 10
            return i + 15
        self.paddings = 1

    def setInput(self):
        if False:
            while True:
                i = 10
        self.dense_x = paddle.randn((1, 5, 6, 8, 3))

class TestKernelSize(TestMaxPool3DFunc):

    def setKernelSize(self):
        if False:
            i = 10
            return i + 15
        self.kernel_sizes = [5, 5, 5]

    def setInput(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(0)
        self.dense_x = paddle.randn((1, 6, 9, 6, 3))

class TestInput(TestMaxPool3DFunc):

    def setInput(self):
        if False:
            print('Hello World!')
        paddle.seed(0)
        self.dense_x = paddle.randn((2, 6, 7, 9, 3))
        dropout = paddle.nn.Dropout(0.8)
        self.dense_x = dropout(self.dense_x)

class TestMaxPool3DAPI(unittest.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        dense_x = paddle.randn((2, 3, 6, 6, 3))
        sparse_x = dense_x.to_sparse_coo(4)
        max_pool3d = paddle.sparse.nn.MaxPool3D(kernel_size=3, data_format='NDHWC')
        out = max_pool3d(sparse_x)
        out = out.to_dense()
        dense_out = paddle.nn.functional.max_pool3d(dense_x, 3, data_format='NDHWC')
        np.testing.assert_allclose(dense_out.numpy(), out.numpy())
if __name__ == '__main__':
    unittest.main()