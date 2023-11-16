import unittest
import numpy as np
import paddle

class TestSparseIsnan(unittest.TestCase):
    """
    Test the API paddle.sparse.isnan on some sparse tensors.
    x: sparse tensor, out: sparse tensor
    """

    def to_sparse(self, x, format):
        if False:
            i = 10
            return i + 15
        if format == 'coo':
            return x.detach().to_sparse_coo(sparse_dim=x.ndim)
        elif format == 'csr':
            return x.detach().to_sparse_csr()

    def check_result(self, x_shape, format, data_type='float32'):
        if False:
            while True:
                i = 10
        raw_inp = np.random.randint(-100, 100, x_shape)
        mask = np.random.randint(0, 2, x_shape)
        inp_x = (raw_inp * mask).astype(data_type)
        inp_x[inp_x > 0] = np.nan
        np_out = np.isnan(inp_x[inp_x != 0])
        dense_x = paddle.to_tensor(inp_x)
        sp_x = self.to_sparse(dense_x, format)
        sp_out = paddle.sparse.isnan(sp_x)
        sp_out_values = sp_out.values().numpy()
        np.testing.assert_allclose(np_out, sp_out_values, rtol=1e-05)

    def test_isnan_shape(self):
        if False:
            print('Hello World!')
        self.check_result([20], 'coo')
        self.check_result([4, 5], 'coo')
        self.check_result([4, 5], 'csr')
        self.check_result([8, 16, 32], 'coo')
        self.check_result([8, 16, 32], 'csr')

    def test_isnan_dtype(self):
        if False:
            i = 10
            return i + 15
        self.check_result([4, 5], 'coo', 'float32')
        self.check_result([4, 5], 'csr', 'float32')
        self.check_result([8, 16, 32], 'coo', 'float64')
        self.check_result([8, 16, 32], 'csr', 'float64')

class TestStatic(unittest.TestCase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        indices = paddle.static.data(name='indices', shape=[2, 3], dtype='int32')
        values = paddle.static.data(name='values', shape=[3], dtype='float32')
        dense_shape = [3, 3]
        sp_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        sp_y = paddle.sparse.isnan(sp_x)
        out = sp_y.to_dense()
        exe = paddle.static.Executor()
        indices_data = [[0, 1, 2], [1, 2, 0]]
        values_data = np.array([1.0, float('nan'), 3.0]).astype('float32')
        fetch = exe.run(feed={'indices': indices_data, 'values': values_data}, fetch_list=[out], return_numpy=True)
        correct_out = np.array([[False, False, False], [False, False, True], [False, False, False]]).astype('float32')
        np.testing.assert_allclose(correct_out, fetch[0], rtol=1e-05)
        paddle.disable_static()
if __name__ == '__main__':
    unittest.main()