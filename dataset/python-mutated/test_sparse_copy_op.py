import unittest
import numpy as np
import paddle

class TestSparseCopy(unittest.TestCase):

    def test_copy_sparse_coo(self):
        if False:
            i = 10
            return i + 15
        np_x = [[0, 1.0, 0], [2.0, 0, 0], [0, 3.0, 0]]
        np_values = [1.0, 2.0, 3.0]
        dense_x = paddle.to_tensor(np_x, dtype='float32')
        coo_x = dense_x.to_sparse_coo(2)
        np_x_2 = [[0, 3.0, 0], [2.0, 0, 0], [0, 3.0, 0]]
        dense_x_2 = paddle.to_tensor(np_x_2, dtype='float32')
        coo_x_2 = dense_x_2.to_sparse_coo(2)
        coo_x_2.copy_(coo_x, True)
        np.testing.assert_array_equal(np_values, coo_x_2.values().numpy())

    def test_copy_sparse_csr(self):
        if False:
            return 10
        np_x = [[0, 1.0, 0], [2.0, 0, 0], [0, 3.0, 0]]
        np_values = [1.0, 2.0, 3.0]
        dense_x = paddle.to_tensor(np_x, dtype='float32')
        csr_x = dense_x.to_sparse_csr()
        np_x_2 = [[0, 3.0, 0], [2.0, 0, 0], [0, 3.0, 0]]
        dense_x_2 = paddle.to_tensor(np_x_2, dtype='float32')
        csr_x_2 = dense_x_2.to_sparse_csr()
        csr_x_2.copy_(csr_x, True)
        np.testing.assert_array_equal(np_values, csr_x_2.values().numpy())