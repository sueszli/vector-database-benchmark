import collections
import unittest
import numpy as np
import paddle
from paddle import nn
from paddle.nn.utils import spectral_norm

class TestDygraphSpectralNorm(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_test_case()
        self.set_data()

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.batch_size = 3
        self.data_desc = (['x', [2, 12, 12]],)
        self.n_power_iterations = 1
        self.eps = 1e-12
        self.dim = None

    def set_data(self):
        if False:
            return 10
        self.data = collections.OrderedDict()
        for desc in self.data_desc:
            data_name = desc[0]
            data_shape = desc[1]
            data_value = np.random.random(size=[self.batch_size] + data_shape).astype('float32')
            self.data[data_name] = data_value

    def spectral_normalize(self, weight, u, v, dim, power_iters, eps):
        if False:
            return 10
        shape = weight.shape
        weight_mat = weight.copy()
        h = shape[dim]
        w = np.prod(shape) // h
        if dim != 0:
            perm = [dim] + [d for d in range(len(shape)) if d != dim]
            weight_mat = weight_mat.transpose(perm)
        weight_mat = weight_mat.reshape((h, w))
        u = u.reshape((h, 1))
        v = v.reshape((w, 1))
        for i in range(power_iters):
            v = np.matmul(weight_mat.T, u)
            v_norm = np.sqrt((v * v).sum())
            v = v / (v_norm + eps)
            u = np.matmul(weight_mat, v)
            u_norm = np.sqrt((u * u).sum())
            u = u / (u_norm + eps)
        sigma = (u * np.matmul(weight_mat, v)).sum()
        return weight / sigma

    def test_check_output(self):
        if False:
            print('Hello World!')
        linear = paddle.nn.Conv2D(2, 1, 3)
        before_weight = linear.weight.numpy().copy()
        if self.dim is None:
            if isinstance(linear, (nn.Conv1DTranspose, nn.Conv2DTranspose, nn.Conv3DTranspose, nn.Linear)):
                self.dim = 1
            else:
                self.dim = 0
        else:
            self.dim = (self.dim + len(before_weight)) % len(before_weight)
        sn = spectral_norm(linear, n_power_iterations=self.n_power_iterations, eps=self.eps, dim=self.dim)
        u = sn.weight_u.numpy().copy()
        v = sn.weight_v.numpy().copy()
        outputs = []
        for (name, data) in self.data.items():
            output = linear(paddle.to_tensor(data))
            outputs.append(output.numpy())
        self.actual_outputs = linear.weight.numpy()
        expect_output = self.spectral_normalize(before_weight, u, v, self.dim, self.n_power_iterations, self.eps)
        for (expect, actual) in zip(expect_output, self.actual_outputs):
            np.testing.assert_allclose(np.array(actual), np.array(expect), rtol=1e-05, atol=0.001)

class TestDygraphWeightNormCase(TestDygraphSpectralNorm):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]],)
        self.n_power_iterations = 1
        self.eps = 1e-12
        self.dim = None

class TestDygraphWeightNormWithIterations(TestDygraphSpectralNorm):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]],)
        self.n_power_iterations = 2
        self.eps = 1e-12
        self.dim = None

class TestDygraphWeightNormWithDim(TestDygraphSpectralNorm):

    def init_test_case(self):
        if False:
            return 10
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]],)
        self.n_power_iterations = 1
        self.eps = 1e-12
        self.dim = 1

class TestDygraphWeightNormWithEps(TestDygraphSpectralNorm):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]],)
        self.n_power_iterations = 1
        self.eps = 1e-10
        self.dim = None
if __name__ == '__main__':
    unittest.main()