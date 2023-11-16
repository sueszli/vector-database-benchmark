import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle
from paddle import _C_ops
from paddle.base.framework import Program, program_guard
paddle.enable_static()

def spectral_norm(weight, u, v, dim, power_iters, eps):
    if False:
        while True:
            i = 10
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

def spectral_norm_wrapper(weight, u, v, dim, power_iters, eps):
    if False:
        return 10
    return _C_ops.spectral_norm(weight, u, v, dim, power_iters, eps)

@skip_check_grad_ci(reason='Spectral norm do not check grad when power_iters > 0 because grad is not calculated in power iterations, which cannot be checked by python grad unittests')
class TestSpectralNormOpNoGrad(OpTest):

    def setUp(self):
        if False:
            return 10
        self.initTestCase()
        self.op_type = 'spectral_norm'
        self.python_api = spectral_norm_wrapper
        weight = np.random.random(self.weight_shape).astype('float64')
        u = np.random.normal(0.0, 1.0, self.u_shape).astype('float64')
        v = np.random.normal(0.0, 1.0, self.v_shape).astype('float64')
        self.attrs = {'dim': self.dim, 'power_iters': self.power_iters, 'eps': self.eps}
        self.inputs = {'Weight': weight, 'U': u, 'V': v}
        output = spectral_norm(weight, u, v, self.dim, self.power_iters, self.eps)
        self.outputs = {'Out': output}

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def initTestCase(self):
        if False:
            for i in range(10):
                print('nop')
        self.weight_shape = (10, 12)
        self.u_shape = (10,)
        self.v_shape = (12,)
        self.dim = 0
        self.power_iters = 5
        self.eps = 1e-12

@skip_check_grad_ci(reason='Spectral norm do not check grad when power_iters > 0 because grad is not calculated in power iterations, which cannot be checked by python grad unittests')
class TestSpectralNormOpNoGrad2(TestSpectralNormOpNoGrad):

    def initTestCase(self):
        if False:
            return 10
        self.weight_shape = (2, 3, 3, 3)
        self.u_shape = (3,)
        self.v_shape = (18,)
        self.dim = 1
        self.power_iters = 10
        self.eps = 1e-12

class TestSpectralNormOp(TestSpectralNormOpNoGrad):

    def test_check_grad_ignore_uv(self):
        if False:
            print('Hello World!')
        self.check_grad(['Weight'], 'Out', no_grad_set={'U', 'V'})

    def initTestCase(self):
        if False:
            print('Hello World!')
        self.weight_shape = (10, 12)
        self.u_shape = (10,)
        self.v_shape = (12,)
        self.dim = 0
        self.power_iters = 0
        self.eps = 1e-12

class TestSpectralNormOp2(TestSpectralNormOp):

    def initTestCase(self):
        if False:
            return 10
        self.weight_shape = (2, 6, 3, 3)
        self.u_shape = (6,)
        self.v_shape = (18,)
        self.dim = 1
        self.power_iters = 0
        self.eps = 1e-12

class TestSpectralNormOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with program_guard(Program(), Program()):

            def test_Variable():
                if False:
                    print('Hello World!')
                weight_1 = np.random.random((2, 4)).astype('float32')
                paddle.static.nn.spectral_norm(weight_1, dim=1, power_iters=2)
            self.assertRaises(TypeError, test_Variable)

            def test_weight_dtype():
                if False:
                    i = 10
                    return i + 15
                weight_2 = np.random.random((2, 4)).astype('int32')
                paddle.static.nn.spectral_norm(weight_2, dim=1, power_iters=2)
            self.assertRaises(TypeError, test_weight_dtype)

            def test_dim_out_of_range_1():
                if False:
                    i = 10
                    return i + 15
                weight_3 = np.random.random((2, 4)).astype('float32')
                tensor_3 = paddle.to_tensor(weight_3)
                paddle.static.nn.spectral_norm(tensor_3, dim=1382376303, power_iters=2)
            self.assertRaises(ValueError, test_dim_out_of_range_1)

            def test_dim_out_of_range_2():
                if False:
                    print('Hello World!')
                weight_4 = np.random.random((2, 4)).astype('float32')
                tensor_4 = paddle.to_tensor(weight_4)
                paddle.static.nn.spectral_norm(tensor_4, dim=-1, power_iters=2)
            self.assertRaises(ValueError, test_dim_out_of_range_2)

class TestDygraphSpectralNormOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with program_guard(Program(), Program()):
            shape = (2, 4, 3, 3)
            spectralNorm = paddle.nn.SpectralNorm(shape, dim=1, power_iters=2)

            def test_Variable():
                if False:
                    i = 10
                    return i + 15
                weight_1 = np.random.random((2, 4)).astype('float32')
                spectralNorm(weight_1)
            self.assertRaises(TypeError, test_Variable)

            def test_weight_dtype():
                if False:
                    i = 10
                    return i + 15
                weight_2 = np.random.random((2, 4)).astype('int32')
                spectralNorm(weight_2)
            self.assertRaises(TypeError, test_weight_dtype)
if __name__ == '__main__':
    unittest.main()