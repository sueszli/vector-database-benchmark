import os
import re
import unittest
import numpy as np
import paddle
paddle.set_default_dtype('float64')

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

class TestAddmm(unittest.TestCase):

    def check_result(self, input_shape, x_shape, y_shape, format):
        if False:
            while True:
                i = 10
        if len(x_shape) == 3:
            mask = paddle.randint(0, 2, [x_shape[-2], x_shape[-1]])
        else:
            mask = paddle.randint(0, 2, x_shape)
        origin_input = paddle.rand(input_shape)
        origin_x = paddle.rand(x_shape) * mask
        origin_y = paddle.rand(y_shape)
        dense_input = origin_input.detach()
        dense_input.stop_gradient = False
        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_y = origin_y.detach()
        dense_y.stop_gradient = False
        dense_out = 2.0 * paddle.matmul(dense_x, dense_y) + 3.0 * dense_input
        sp_input = dense_input.detach()
        sp_input.stop_gradient = False
        if format == 'coo':
            sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
        else:
            sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_y = origin_y.detach()
        sp_y.stop_gradient = False
        sp_out = paddle.sparse.addmm(sp_input, sp_x, sp_y, 3.0, 2.0)
        np.testing.assert_allclose(sp_out.numpy(), dense_out.numpy(), rtol=1e-05)
        if get_cuda_version() >= 11030:
            dense_out.backward()
            sp_out.backward()
            np.testing.assert_allclose(sp_input.grad.numpy(), dense_input.grad.numpy(), rtol=1e-05)
            np.testing.assert_allclose(sp_x.grad.to_dense().numpy(), (dense_x.grad * mask).numpy(), rtol=1e-05)
            np.testing.assert_allclose(sp_y.grad.numpy(), dense_y.grad.numpy(), rtol=1e-05)

    @unittest.skipIf(not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000, 'only support cuda>=11.0')
    def test_addmm_2d(self):
        if False:
            return 10
        self.check_result([16, 10], [16, 12], [12, 10], 'coo')
        self.check_result([16, 10], [16, 12], [12, 10], 'csr')

    @unittest.skipIf(not paddle.is_compiled_with_cuda() or get_cuda_version() < 11080, 'only support cuda>=11.8')
    def test_addmm_3d(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_result([8, 16, 10], [8, 16, 12], [8, 12, 10], 'coo')
        self.check_result([8, 16, 10], [8, 16, 12], [8, 12, 10], 'csr')
if __name__ == '__main__':
    unittest.main()