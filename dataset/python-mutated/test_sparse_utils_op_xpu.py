import unittest
import numpy as np
import paddle

class TestSparseCreate(unittest.TestCase):

    def test_create_coo_by_tensor(self):
        if False:
            print('Hello World!')
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        dense_indices = paddle.to_tensor(indices)
        dense_elements = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(dense_indices, dense_elements, dense_shape, stop_gradient=False)
        np.testing.assert_array_equal(indices, coo.indices().numpy())
        np.testing.assert_array_equal(values, coo.values().numpy())

    def test_create_coo_by_np(self):
        if False:
            return 10
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        np.testing.assert_array_equal(3, coo.nnz())
        np.testing.assert_array_equal(indices, coo.indices().numpy())
        np.testing.assert_array_equal(values, coo.values().numpy())

    def test_place(self):
        if False:
            print('Hello World!')
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        dense_shape = [2, 2]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        assert coo.place.is_xpu_place()
        assert coo.values().place.is_xpu_place()
        assert coo.indices().place.is_xpu_place()

    def test_dtype(self):
        if False:
            return 10
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        dense_shape = [2, 2]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, dtype='float64')
        assert coo.dtype == paddle.float64

    def test_create_coo_no_shape(self):
        if False:
            print('Hello World!')
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(indices, values)
        assert [2, 2] == coo.shape
if __name__ == '__main__':
    unittest.main()