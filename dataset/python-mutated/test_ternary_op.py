import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error
from .test_grad import ngrad
from .test_cuda import test_cuda

class TestTernaryOp(unittest.TestCase):

    def test_with_np(self):
        if False:
            return 10
        np.random.seed(0)
        a = np.random.rand(5, 10).astype('float32')
        b = np.random.rand(5, 10).astype('float32')
        ja = jt.array(a)
        jb = jt.array(b)
        jc = jt.ternary(ja > jb, ja, jb)
        assert (jc.data == np.maximum(a, b)).all(), f'\n{jc.data}\n{np.maximum(a, b)}\n{a}\n{b}'
        (jda, jdb) = jt.grad(jc, [ja, jb])
        assert (jda.data == (a > b) * 1).all()
        assert (jdb.data == 1 - (a > b)).all()

    def test_where(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        a = np.random.rand(5, 10).astype('float32')
        b = np.random.rand(5, 10).astype('float32')
        ja = jt.array(a)
        jb = jt.array(b)
        jc = jt.where(ja > jb, ja, jb)
        assert (jc.data == np.maximum(a, b)).all(), f'\n{jc.data}\n{np.maximum(a, b)}\n{a}\n{b}'
        (jda, jdb) = jt.grad(jc, [ja, jb])
        assert (jda.data == (a > b) * 1).all()
        assert (jdb.data == 1 - (a > b)).all()

    def test_min(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(1)
        a = np.random.rand(5, 10).astype('float32')
        b = np.random.rand(5, 10).astype('float32')
        ja = jt.array(a)
        jb = jt.array(b)
        jc = jt.minimum(ja, jb)
        assert (jc.data == np.minimum(a, b)).all(), f'\n{jc.data}\n{np.minimum(a, b)}\n{a}\n{b}'
        (jda, jdb) = jt.grad(jc, [ja, jb])
        assert (jda.data == (a < b) * 1).all()
        assert (jdb.data == 1 - (a < b)).all()

class TestTernaryOpCuda(TestTernaryOp, test_cuda(2)):
    pass
if __name__ == '__main__':
    unittest.main()