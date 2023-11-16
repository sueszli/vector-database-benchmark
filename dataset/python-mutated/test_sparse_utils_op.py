import unittest
import numpy as np
import paddle
from paddle.base import core
devices = ['cpu', 'gpu']

class TestSparseCreate(unittest.TestCase):

    def test_create_coo_by_tensor(self):
        if False:
            for i in range(10):
                print('nop')
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

    def test_create_csr_by_tensor(self):
        if False:
            while True:
                i = 10
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        dense_crows = paddle.to_tensor(crows)
        dense_cols = paddle.to_tensor(cols)
        dense_elements = paddle.to_tensor(values, dtype='float32')
        stop_gradient = False
        csr = paddle.sparse.sparse_csr_tensor(dense_crows, dense_cols, dense_elements, dense_shape, stop_gradient=stop_gradient)

    def test_create_csr_by_np(self):
        if False:
            print('Hello World!')
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        np.testing.assert_array_equal(5, csr.nnz())
        np.testing.assert_array_equal(crows, csr.crows().numpy())
        np.testing.assert_array_equal(cols, csr.cols().numpy())
        np.testing.assert_array_equal(values, csr.values().numpy())

    def test_place(self):
        if False:
            print('Hello World!')
        place = core.CPUPlace()
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        dense_shape = [2, 2]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, place=place)
        assert coo.place.is_cpu_place()
        assert coo.values().place.is_cpu_place()
        assert coo.indices().place.is_cpu_place()
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, [3, 5], place=place)
        assert csr.place.is_cpu_place()
        assert csr.crows().place.is_cpu_place()
        assert csr.cols().place.is_cpu_place()
        assert csr.values().place.is_cpu_place()

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
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, [3, 5], dtype='float16')
        assert csr.dtype == paddle.float16

    def test_create_coo_no_shape(self):
        if False:
            return 10
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(indices, values)
        assert [2, 2] == coo.shape

class TestSparseConvert(unittest.TestCase):

    def test_to_sparse_coo(self):
        if False:
            while True:
                i = 10
        x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        dense_x = paddle.to_tensor(x, dtype='float32', stop_gradient=False)
        out = dense_x.to_sparse_coo(2)
        np.testing.assert_array_equal(out.indices().numpy(), indices)
        np.testing.assert_array_equal(out.values().numpy(), values)
        out_grad_indices = [[0, 1], [0, 1]]
        out_grad_values = [2.0, 3.0]
        out_grad = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(out_grad_indices), paddle.to_tensor(out_grad_values), shape=out.shape, stop_gradient=True)
        out.backward(out_grad)
        np.testing.assert_array_equal(dense_x.grad.numpy(), out_grad.to_dense().numpy())

    def test_coo_to_dense(self):
        if False:
            while True:
                i = 10
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        indices_dtypes = ['int32', 'int64']
        for indices_dtype in indices_dtypes:
            sparse_x = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(indices, dtype=indices_dtype), paddle.to_tensor(values), shape=[3, 4], stop_gradient=False)
            sparse_x.retain_grads()
            dense_tensor = sparse_x.to_dense()
            out_grad = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
            dense_tensor.backward(paddle.to_tensor(out_grad))
            correct_x_grad = [2.0, 4.0, 7.0, 9.0, 10.0]
            np.testing.assert_array_equal(correct_x_grad, sparse_x.grad.values().numpy())
            paddle.device.set_device('cpu')
            sparse_x_cpu = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(indices, dtype=indices_dtype), paddle.to_tensor(values), shape=[3, 4], stop_gradient=False)
            sparse_x_cpu.retain_grads()
            dense_tensor_cpu = sparse_x_cpu.to_dense()
            dense_tensor_cpu.backward(paddle.to_tensor(out_grad))
            np.testing.assert_array_equal(correct_x_grad, sparse_x_cpu.grad.values().numpy())

    def test_to_sparse_csr(self):
        if False:
            return 10
        x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_x = paddle.to_tensor(x)
        out = dense_x.to_sparse_csr()
        np.testing.assert_array_equal(out.crows().numpy(), crows)
        np.testing.assert_array_equal(out.cols().numpy(), cols)
        np.testing.assert_array_equal(out.values().numpy(), values)
        dense_tensor = out.to_dense()
        np.testing.assert_array_equal(dense_tensor.numpy(), x)

    def test_coo_values_grad(self):
        if False:
            return 10
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        sparse_x = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(indices), paddle.to_tensor(values), shape=[3, 4], stop_gradient=False)
        sparse_x.retain_grads()
        values_tensor = sparse_x.values()
        out_grad = [2.0, 3.0, 5.0, 8.0, 9.0]
        values_tensor.backward(paddle.to_tensor(out_grad))
        np.testing.assert_array_equal(out_grad, sparse_x.grad.values().numpy())
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
        sparse_x = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(indices), paddle.to_tensor(values), shape=[3, 4, 2], stop_gradient=False)
        sparse_x.retain_grads()
        values_tensor = sparse_x.values()
        out_grad = [[2.0, 2.0], [3.0, 3.0], [5.0, 5.0], [8.0, 8.0], [9.0, 9.0]]
        values_tensor.backward(paddle.to_tensor(out_grad))
        np.testing.assert_array_equal(out_grad, sparse_x.grad.values().numpy())

    def test_sparse_coo_tensor_grad(self):
        if False:
            return 10
        for device in devices:
            if device == 'cpu' or (device == 'gpu' and paddle.is_compiled_with_cuda()):
                paddle.device.set_device(device)
                indices = [[0, 1], [0, 1]]
                values = [1, 2]
                indices = paddle.to_tensor(indices, dtype='int32')
                values = paddle.to_tensor(values, dtype='float32', stop_gradient=False)
                sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, shape=[2, 2], stop_gradient=False)
                grad_indices = [[0, 1], [1, 1]]
                grad_values = [2, 3]
                grad_indices = paddle.to_tensor(grad_indices, dtype='int32')
                grad_values = paddle.to_tensor(grad_values, dtype='float32')
                sparse_out_grad = paddle.sparse.sparse_coo_tensor(grad_indices, grad_values, shape=[2, 2])
                sparse_x.backward(sparse_out_grad)
                correct_values_grad = [0, 3]
                np.testing.assert_array_equal(correct_values_grad, values.grad.numpy())
                values = [[1, 1], [2, 2]]
                values = paddle.to_tensor(values, dtype='float32', stop_gradient=False)
                sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, shape=[2, 2, 2], stop_gradient=False)
                grad_values = [[2, 2], [3, 3]]
                grad_values = paddle.to_tensor(grad_values, dtype='float32')
                sparse_out_grad = paddle.sparse.sparse_coo_tensor(grad_indices, grad_values, shape=[2, 2, 2])
                sparse_x.backward(sparse_out_grad)
                correct_values_grad = [[0, 0], [3, 3]]
                np.testing.assert_array_equal(correct_values_grad, values.grad.numpy())

    def test_sparse_coo_tensor_sorted(self):
        if False:
            print('Hello World!')
        for device in devices:
            if device == 'cpu' or (device == 'gpu' and paddle.is_compiled_with_cuda()):
                paddle.device.set_device(device)
                indices = [[1, 0, 0], [0, 1, 1]]
                values = [1.0, 2.0, 3.0]
                indices = paddle.to_tensor(indices, dtype='int32')
                values = paddle.to_tensor(values, dtype='float32')
                sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)
                sparse_x = paddle.sparse.coalesce(sparse_x)
                indices_sorted = [[0, 1], [1, 0]]
                values_sorted = [5.0, 1.0]
                np.testing.assert_array_equal(indices_sorted, sparse_x.indices().numpy())
                np.testing.assert_array_equal(values_sorted, sparse_x.values().numpy())
                values = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
                values = paddle.to_tensor(values, dtype='float32')
                sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)
                sparse_x = paddle.sparse.coalesce(sparse_x)
                values_sorted = [[5.0, 5.0], [1.0, 1.0]]
                np.testing.assert_array_equal(indices_sorted, sparse_x.indices().numpy())
                np.testing.assert_array_equal(values_sorted, sparse_x.values().numpy())

    def test_batch_csr(self):
        if False:
            while True:
                i = 10

        def verify(dense_x):
            if False:
                for i in range(10):
                    print('nop')
            sparse_x = dense_x.to_sparse_csr()
            out = sparse_x.to_dense()
            np.testing.assert_allclose(out.numpy(), dense_x.numpy())
        shape = np.random.randint(low=1, high=10, size=3)
        shape = list(shape)
        dense_x = paddle.randn(shape)
        dense_x = paddle.nn.functional.dropout(dense_x, p=0.5)
        verify(dense_x)
        shape[0] = 1
        dense_x = paddle.randn(shape)
        dense_x = paddle.nn.functional.dropout(dense_x, p=0.5)
        verify(dense_x)
        shape = np.random.randint(low=3, high=10, size=3)
        shape = list(shape)
        dense_x = paddle.randn(shape)
        dense_x[0] = 0
        verify(dense_x)
        dense_x = paddle.randn(shape)
        dense_x[1] = 0
        verify(dense_x)
        dense_x = paddle.randn(shape)
        dense_x[2] = 0
        verify(dense_x)

    def test_zero_nnz(self):
        if False:
            for i in range(10):
                print('nop')
        for device in devices:
            if device == 'cpu' or (device == 'gpu' and paddle.is_compiled_with_cuda()):
                paddle.device.set_device(device)
                x1 = paddle.zeros([2, 2, 2])
                x2 = paddle.zeros([2, 2, 2])
                sp_csr_x = x1.to_sparse_csr()
                sp_coo_x = x2.to_sparse_coo(1)
                sp_coo_x.stop_gradient = False
                out1 = sp_csr_x.to_dense()
                out2 = sp_coo_x.to_dense()
                out2.backward()
                np.testing.assert_allclose(out1.numpy(), x1.numpy())
                np.testing.assert_allclose(out2.numpy(), x2.numpy())
                np.testing.assert_allclose(sp_coo_x.grad.to_dense().numpy().sum(), 0.0)

class TestCooError(unittest.TestCase):

    def test_small_shape(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            indices = [[2, 3], [0, 2]]
            values = [1, 2]
            dense_shape = [2, 2]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, shape=dense_shape)

    def test_same_nnz(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            indices = [[1, 2], [1, 0]]
            values = [1, 2, 3]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)

    def test_same_dimensions(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            indices = [[1, 2], [1, 0]]
            values = [1, 2, 3]
            shape = [2, 3, 4]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, shape=shape)

    def test_indices_dtype(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            indices = [[1.0, 2.0], [0, 1]]
            values = [1, 2]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)

class TestCsrError(unittest.TestCase):

    def test_dimension1(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3]
            cols = [0, 1, 2]
            values = [1, 2, 3]
            shape = [3]
            sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values, shape)

    def test_dimension2(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3]
            cols = [0, 1, 2]
            values = [1, 2, 3]
            shape = [3, 3, 3, 3]
            sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values, shape)

    def test_same_shape1(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3]
            cols = [0, 1, 2, 3]
            values = [1, 2, 3]
            shape = [3, 4]
            sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values, shape)

    def test_same_shape2(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3]
            cols = [0, 1, 2, 3]
            values = [1, 2, 3, 4]
            shape = [3, 4]
            sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values, shape)

    def test_same_shape3(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3, 0, 1, 2]
            cols = [0, 1, 2, 3, 0, 1, 2]
            values = [1, 2, 3, 4, 0, 1, 2]
            shape = [2, 3, 4]
            sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values, shape)

    def test_crows_first_value(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            crows = [1, 1, 2, 3]
            cols = [0, 1, 2]
            values = [1, 2, 3]
            shape = [3, 4]
            sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values, shape)

    def test_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            crows = [0, 1, 2, 3.0]
            cols = [0, 1, 2]
            values = [1, 2, 3]
            shape = [3]
            sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values, shape)
if __name__ == '__main__':
    unittest.main()