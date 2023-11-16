import os
import re
import unittest
import numpy as np
import paddle
paddle.seed(100)

def get_cuda_version():
    if False:
        return 10
    result = os.popen('nvcc --version').read()
    regex = 'release (\\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        (integer, decimal) = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1

@unittest.skipIf(not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000, 'paddle is not compiled with CUDA and cuda version need to >= 11.0')
class TestCsrMv(unittest.TestCase):

    def test_mv(self):
        if False:
            return 10
        paddle.set_default_dtype('float64')
        origin_x = paddle.rand([64, 32])
        mask = paddle.randint(0, 2, [64, 32])
        origin_x = origin_x * mask
        origin_vec = paddle.rand([32])
        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_vec = origin_vec.detach()
        dense_vec.stop_gradient = False
        dense_out = paddle.mv(dense_x, dense_vec)
        dense_out.backward()
        sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_vec = origin_vec.detach()
        sp_vec.stop_gradient = False
        sp_out = paddle.sparse.mv(sp_x, sp_vec)
        sp_out.backward()
        np.testing.assert_allclose(sp_out.numpy(), dense_out.numpy(), rtol=1e-05)
        np.testing.assert_allclose(sp_x.grad.to_dense().numpy(), (dense_x.grad * mask).numpy(), rtol=1e-05)
        np.testing.assert_allclose(sp_vec.grad.numpy(), dense_vec.grad.numpy(), rtol=1e-05)

@unittest.skipIf(not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000, 'paddle is not compiled with CUDA and cuda version need to >= 11.0')
class TestCooMv(unittest.TestCase):

    def test_mv(self):
        if False:
            while True:
                i = 10
        paddle.set_default_dtype('float64')
        origin_x = paddle.rand([64, 32])
        mask = paddle.randint(0, 2, [64, 32])
        origin_x = origin_x * mask
        origin_vec = paddle.rand([32])
        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_vec = origin_vec.detach()
        dense_vec.stop_gradient = False
        dense_out = paddle.mv(dense_x, dense_vec)
        dense_out.backward()
        sp_x = origin_x.detach().to_sparse_coo(sparse_dim=2)
        sp_x.stop_gradient = False
        sp_vec = origin_vec.detach()
        sp_vec.stop_gradient = False
        sp_out = paddle.sparse.mv(sp_x, sp_vec)
        sp_out.backward()
        np.testing.assert_allclose(sp_out.numpy(), dense_out.numpy(), rtol=1e-05)
        np.testing.assert_allclose(sp_x.grad.to_dense().numpy(), (dense_x.grad * mask).numpy(), rtol=1e-05)
        np.testing.assert_allclose(sp_vec.grad.numpy(), dense_vec.grad.numpy(), rtol=1e-05)
if __name__ == '__main__':
    unittest.main()