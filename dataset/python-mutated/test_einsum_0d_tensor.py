import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
import paddle
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

class Test0DCase0(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()

    def test_func(self):
        if False:
            return 10
        x = paddle.rand([])
        x.stop_gradient = False
        y = paddle.rand([])
        y.stop_gradient = False
        z = paddle.einsum('...,...->...', x, y)
        assert_allclose(z.numpy(), np.einsum('...,...->...', x.numpy(), y.numpy()), atol=1e-06)
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == []
        assert y.grad.shape == []

class Test0DCase1(Test0DCase0):

    def test_func(self):
        if False:
            while True:
                i = 10
        x = paddle.rand([])
        x.stop_gradient = False
        y = paddle.rand([2, 2])
        y.stop_gradient = False
        z = paddle.einsum('...,ij->...', x, y)
        assert_allclose(z.numpy(), np.einsum('...,ij->...', x.numpy(), y.numpy()), atol=1e-06)
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == []
        assert y.grad.shape == [2, 2]

class Test0DCase2(Test0DCase0):

    def test_func(self):
        if False:
            i = 10
            return i + 15
        x = paddle.rand([2, 2])
        x.stop_gradient = False
        y = paddle.rand([2, 2])
        y.stop_gradient = False
        z = paddle.einsum('ij,ij->', x, y)
        assert_allclose(z.numpy(), np.einsum('ij,ij->', x.numpy(), y.numpy()), atol=1e-06)
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == [2, 2]
        assert y.grad.shape == [2, 2]

class Test0DCase3(Test0DCase0):

    def test_func(self):
        if False:
            while True:
                i = 10
        x = paddle.rand([2, 2])
        x.stop_gradient = True
        y = paddle.rand([2, 2])
        y.stop_gradient = False
        z = paddle.einsum('ij,ij->', x, y)
        assert_allclose(z.numpy(), np.einsum('ij,ij->', x.numpy(), y.numpy()), atol=1e-06)
        z.mean().backward()
        assert z.shape == []
        assert x.grad is None
        assert y.grad.shape == [2, 2]

class Test0DCase4(Test0DCase0):

    def test_func(self):
        if False:
            while True:
                i = 10
        x = paddle.rand([])
        x.stop_gradient = False
        z = paddle.einsum('...->...', x)
        assert_allclose(z.numpy(), np.einsum('...->...', x.numpy()), atol=1e-06)
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == []
        assert x.grad.numpy() == 1.0

class Test0DCase5(Test0DCase0):

    def test_func(self):
        if False:
            i = 10
            return i + 15
        x = paddle.rand([2, 2])
        x.stop_gradient = False
        y = paddle.rand([2, 2])
        y.stop_gradient = False
        z = paddle.einsum('i...j, i...j->...', x, y)
        assert_allclose(z.numpy(), np.einsum('i...j, i...j->...', x.numpy(), y.numpy()), atol=1e-06)
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == [2, 2]
        assert y.grad.shape == [2, 2]

class Test0DCase6(Test0DCase0):

    def test_func(self):
        if False:
            i = 10
            return i + 15
        x = paddle.rand([2, 2])
        x.stop_gradient = False
        z = paddle.einsum('ij->', x)
        assert_allclose(z.numpy(), np.einsum('ij->', x.numpy()), atol=1e-06)
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == [2, 2]

class Test0DCase7(Test0DCase0):

    def test_func(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        3 operands.\n        '
        x = paddle.rand([2, 2])
        y = paddle.rand([])
        z = paddle.rand([])
        x.stop_gradient = False
        y.stop_gradient = False
        z.stop_gradient = False
        o = paddle.einsum('ij...,...,...->...', x, y, z)
        assert_allclose(o.numpy(), np.einsum('ij...,...,...->...', x.numpy(), y.numpy(), z.numpy()), atol=1e-06)
        o.mean().backward()
        assert o.shape == []
        assert x.grad.shape == [2, 2]
        assert y.grad.shape == []
        assert z.grad.shape == []

class Test0DCase8(Test0DCase0):

    def test_func(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        3 operands.\n        '
        x = paddle.rand([2, 2])
        y = paddle.rand([])
        z = paddle.rand([])
        e = paddle.rand([3, 1])
        x.stop_gradient = False
        y.stop_gradient = False
        z.stop_gradient = False
        e.stop_gradient = False
        o = paddle.einsum('ij...,...,..., km->...', x, y, z, e)
        assert_allclose(o.numpy(), np.einsum('ij...,...,...,km->...', x.numpy(), y.numpy(), z.numpy(), e.numpy()), atol=1e-06)
        o.mean().backward()
        assert o.shape == []
        assert x.grad.shape == [2, 2]
        assert y.grad.shape == []
        assert z.grad.shape == []
        assert e.grad.shape == [3, 1]
if __name__ == '__main__':
    unittest.main()