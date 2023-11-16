import unittest
import numpy as np
import paddle
data_5d = [[[2, 3, 4, 5, 6], [0, 1, 2, 4], [0, 1, 2, -4], [3, 3, 4, -2]]]
data_4d = [[[2, 3, 4, 5], [0, 1, 2, 3], [0, 1, 2, -4], [3, 3, 4, -2]]]
data_3d = [[[4, 4, 5], [-3, -2, -1], [1, -3, 2], [3, 3, 4]], [[4, 4, 5], [0, 1, 2], [0, 1, 2], [3, 3, 4]], [[4, 4, 5], [-1], [0], [2]], [[4, 4, 5], [0], [1], [2]], [[4, 4, 5], [1], [2], [3]], [[4, 4, 5], [1, 2], [2, 2], [3, 4]], [[4, 4, 5], [0, 2], [2, 2], [3, 4]]]
data_2d = [[[3, 4], [0], [0], [2]], [[3, 4], [1], [-3], [2]], [[3, 4], [-2, -1], [-3, 0], [2, -1]], [[78, 78], [0, -1], [32, 58], [-2, -1]]]
devices = ['cpu']
if paddle.device.get_device() != 'cpu':
    devices.append(paddle.device.get_device())

class TestSparseSlice(unittest.TestCase):
    """
    Test the API paddle.sparse.slice on some sparse tensors.
    x: sparse, out: sparse
    """

    def _check_result(self, np_x, axes, starts, ends, format='coo'):
        if False:
            return 10
        for device in devices:
            paddle.device.set_device(device)
            self._check_result_with_place(np_x, axes, starts, ends, format)

    def _check_result_with_place(self, np_x, axes, starts, ends, format='coo'):
        if False:
            return 10
        x_shape = np_x.shape
        dense_x = paddle.to_tensor(np_x)
        dense_x.stop_gradient = False
        dense_out = paddle.slice(dense_x, axes, starts, ends)
        if format == 'coo':
            sp_x = paddle.to_tensor(np_x).to_sparse_coo(len(x_shape))
        else:
            sp_x = paddle.to_tensor(np_x).to_sparse_csr()
        sp_x.stop_gradient = False
        sp_out = paddle.sparse.slice(sp_x, axes, starts, ends)
        np.testing.assert_allclose(sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05)
        dense_out.backward()
        sp_out.backward()
        np.testing.assert_allclose(sp_x.grad.to_dense().numpy(), dense_x.grad.numpy() * np_x.astype('bool').astype('int'), rtol=1e-05)

    def check_result_with_shape(self, x_shape, axes, starts, ends, format='coo'):
        if False:
            print('Hello World!')
        mask = np.random.randint(0, 2, x_shape)
        np_x = np.random.randint(-100, 100, x_shape) * mask
        self._check_result(np_x, axes, starts, ends, format)

    def check_result_with_list(self, x, axes, starts, ends, format='coo'):
        if False:
            while True:
                i = 10
        np_x = np.array(x)
        self._check_result(np_x, axes, starts, ends, format)

    def test_coo_5d(self):
        if False:
            for i in range(10):
                print('nop')
        for item in data_5d:
            self.check_result_with_shape(*item, format='coo')

    def test_coo_4d(self):
        if False:
            while True:
                i = 10
        for item in data_4d:
            self.check_result_with_shape(*item, format='coo')

    def test_coo_3d(self):
        if False:
            while True:
                i = 10
        for item in data_3d:
            self.check_result_with_shape(*item, format='coo')

    def test_coo_2d(self):
        if False:
            return 10
        x = [[1, 2, 3, 4], [0, 1, 2, 0]]
        self.check_result_with_list(x, [0, 1], [0, 1], [2, 3], format='coo')
        for item in data_2d:
            self.check_result_with_shape(*item, format='coo')

    def test_coo_1d(self):
        if False:
            for i in range(10):
                print('nop')
        x = [-49, 55, -5, 0, 3, 0, 0, -60, -21, 0, 0, 0]
        self.check_result_with_list(x, [0], [3], [5], format='coo')

    def test_coo_1d_zero(self):
        if False:
            for i in range(10):
                print('nop')
        x = [-49, 55, -5, 0, 3, 0, 0, -60, -21, 0, 0, 0]
        self.check_result_with_list(x, [0], [-3], [-1], format='coo')

    def test_csr_3d(self):
        if False:
            print('Hello World!')
        for item in data_3d:
            self.check_result_with_shape(*item, format='csr')

    def test_csr_3d_zero(self):
        if False:
            return 10
        x = [[[0, 0, 1, 2], [0, 0, 0, 2]]]
        self.check_result_with_list(x, [1, 2], [0, 0], [2, 2], format='csr')

    def test_csr_2d(self):
        if False:
            i = 10
            return i + 15
        for item in data_2d:
            self.check_result_with_shape(*item, format='csr')

    def test_csr_2d_zero(self):
        if False:
            return 10
        x = [[0, 0, 1, 2], [0, 0, 0, 1]]
        self.check_result_with_list(x, [0, 1], [0, 0], [2, 2], format='csr')

class TestSparseCooSliceStatic(unittest.TestCase):

    def _check_result_coo(self, np_x, axes, starts, ends):
        if False:
            for i in range(10):
                print('nop')
        for device in devices:
            paddle.device.set_device(device)
            self._check_result_coo_with_place(np_x, axes, starts, ends)

    def _check_result_coo_with_place(self, np_x, axes, starts, ends):
        if False:
            for i in range(10):
                print('nop')
        x_shape = np_x.shape
        dense_x = paddle.to_tensor(np_x)
        dense_x.stop_gradient = False
        dense_out = paddle.slice(dense_x, axes, starts, ends)
        sp_x = paddle.to_tensor(np_x).to_sparse_coo(len(x_shape))
        indices_data = sp_x.detach().indices()
        values_data = sp_x.detach().values()
        paddle.enable_static()
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            indices = paddle.static.data(name='indices', shape=indices_data.shape, dtype=indices_data.dtype)
            values = paddle.static.data(name='values', shape=values_data.shape, dtype=values_data.dtype)
            sp_x = paddle.sparse.sparse_coo_tensor(indices, values, shape=dense_x.shape, dtype=dense_x.dtype)
            sp_out = paddle.sparse.slice(sp_x, axes, starts, ends)
            sp_dense_out = sp_out.to_dense()
            exe = paddle.static.Executor()
            res = exe.run(feed={'indices': indices_data.numpy(), 'values': values_data.numpy()}, fetch_list=[sp_dense_out], return_numpy=True)
            np.testing.assert_allclose(dense_out.numpy(), res[0], rtol=1e-05)
        paddle.disable_static()

    def check_result_with_shape(self, x_shape, axes, starts, ends, format='coo'):
        if False:
            print('Hello World!')
        mask = np.random.randint(0, 2, x_shape)
        np_x = np.random.randint(-100, 100, x_shape) * mask
        if format == 'coo':
            self._check_result_coo(np_x, axes, starts, ends)

    def check_result_with_list(self, x, axes, starts, ends, format='coo'):
        if False:
            i = 10
            return i + 15
        np_x = np.array(x)
        if format == 'coo':
            self._check_result_coo(np_x, axes, starts, ends)

    def test_coo_5d(self):
        if False:
            for i in range(10):
                print('nop')
        for item in data_5d:
            self.check_result_with_shape(*item, format='coo')

    def test_coo_4d(self):
        if False:
            i = 10
            return i + 15
        for item in data_4d:
            self.check_result_with_shape(*item, format='coo')

    def test_coo_3d(self):
        if False:
            for i in range(10):
                print('nop')
        for item in data_3d:
            self.check_result_with_shape(*item, format='coo')

    def test_coo_2d(self):
        if False:
            print('Hello World!')
        for item in data_2d:
            self.check_result_with_shape(*item, format='coo')

    def test_coo_1d(self):
        if False:
            while True:
                i = 10
        x = [-49, 55, -5, 0, 3, 0, 0, -60, -21, 0, 0, 0]
        self.check_result_with_list(x, [0], [3], [5], format='coo')

    def test_coo_1d_zero(self):
        if False:
            for i in range(10):
                print('nop')
        x = [-49, 55, -5, 0, 3, 0, 0, -60, -21, 0, 0, 0]
        self.check_result_with_list(x, [0], [-3], [-1], format='coo')
if __name__ == '__main__':
    unittest.main()