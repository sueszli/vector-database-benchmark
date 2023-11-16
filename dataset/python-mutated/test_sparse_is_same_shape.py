import unittest
import paddle
from paddle.sparse.binary import is_same_shape

class TestSparseIsSameShapeAPI(unittest.TestCase):
    """
    test paddle.sparse.is_same_shape
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.shapes = [[2, 5, 8], [3, 4]]
        self.tensors = [paddle.rand(self.shapes[0]), paddle.rand(self.shapes[0]), paddle.rand(self.shapes[1])]
        self.sparse_dim = 2

    def test_dense_dense(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(is_same_shape(self.tensors[0], self.tensors[1]))
        self.assertFalse(is_same_shape(self.tensors[0], self.tensors[2]))
        self.assertFalse(is_same_shape(self.tensors[1], self.tensors[2]))

    def test_dense_csr(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(is_same_shape(self.tensors[0], self.tensors[1].to_sparse_csr()))
        self.assertFalse(is_same_shape(self.tensors[0], self.tensors[2].to_sparse_csr()))
        self.assertFalse(is_same_shape(self.tensors[1], self.tensors[2].to_sparse_csr()))

    def test_dense_coo(self):
        if False:
            return 10
        self.assertTrue(is_same_shape(self.tensors[0], self.tensors[1].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(is_same_shape(self.tensors[0], self.tensors[2].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(is_same_shape(self.tensors[1], self.tensors[2].to_sparse_coo(self.sparse_dim)))

    def test_csr_dense(self):
        if False:
            print('Hello World!')
        self.assertTrue(is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[1]))
        self.assertFalse(is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[2]))
        self.assertFalse(is_same_shape(self.tensors[1].to_sparse_csr(), self.tensors[2]))

    def test_csr_csr(self):
        if False:
            return 10
        self.assertTrue(is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[1].to_sparse_csr()))
        self.assertFalse(is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[2].to_sparse_csr()))
        self.assertFalse(is_same_shape(self.tensors[1].to_sparse_csr(), self.tensors[2].to_sparse_csr()))

    def test_csr_coo(self):
        if False:
            print('Hello World!')
        self.assertTrue(is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[1].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[2].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(is_same_shape(self.tensors[1].to_sparse_csr(), self.tensors[2].to_sparse_coo(self.sparse_dim)))

    def test_coo_dense(self):
        if False:
            while True:
                i = 10
        self.assertTrue(is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim), self.tensors[1]))
        self.assertFalse(is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim), self.tensors[2]))
        self.assertFalse(is_same_shape(self.tensors[1].to_sparse_coo(self.sparse_dim), self.tensors[2]))

    def test_coo_csr(self):
        if False:
            while True:
                i = 10
        self.assertTrue(is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim), self.tensors[1].to_sparse_csr()))
        self.assertFalse(is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim), self.tensors[2].to_sparse_csr()))
        self.assertFalse(is_same_shape(self.tensors[1].to_sparse_coo(self.sparse_dim), self.tensors[2].to_sparse_csr()))

    def test_coo_coo(self):
        if False:
            while True:
                i = 10
        self.assertTrue(is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim), self.tensors[1].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim), self.tensors[2].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(is_same_shape(self.tensors[1].to_sparse_coo(self.sparse_dim), self.tensors[2].to_sparse_coo(self.sparse_dim)))
if __name__ == '__main__':
    unittest.main()