import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error
from .test_grad import ngrad

def pool(x, size, op):
    if False:
        for i in range(10):
            print('nop')
    (N, H, W, C) = x.shape
    h = (H + size - 1) // size
    w = (W + size - 1) // size
    return x.reindex_reduce(op, [N, h, w, C], ['i0', f'i1/{size}', f'i2/{size}', 'i3'])

def pool_naive(x, size, op):
    if False:
        for i in range(10):
            print('nop')
    (N, H, W, C) = x.shape
    h = (H + size - 1) // size
    w = (W + size - 1) // size
    y = np.zeros([N, h, w, C], dtype='float64')
    x = np.float64(x)
    if op == 'maximum':
        y[:] = -1e+100
        fop = lambda x, y: np.maximum(x, y)
    elif op == 'minimum':
        y[:] = 1e+100
        fop = lambda x, y: np.minimum(x, y)
    elif op == 'multiply':
        y[:] = 1
        fop = lambda x, y: x * y
    else:
        assert op == 'add'
        fop = lambda x, y: x + y
    for i0 in range(N):
        for i1 in range(H):
            for i2 in range(W):
                for i3 in range(C):
                    y[i0, i1 // size, i2 // size, i3] = fop(y[i0, i1 // size, i2 // size, i3], x[i0, i1, i2, i3])
    return y
ops = ['maximum', 'minimum', 'multiply', 'add']

class TestReindexReduceOp(unittest.TestCase):

    def test_pool(self):
        if False:
            for i in range(10):
                print('nop')
        (N, H, W, C) = (3, 10, 10, 4)
        size = 3
        for op in ops:
            x = jt.random([N, H, W, C])
            y = pool(x, size, op)
            ny = pool_naive(x.data, size, op)
            assert np.allclose(y.data, ny), (op, y.data, ny)

    def test_pool_grad(self):
        if False:
            i = 10
            return i + 15
        jt.set_seed(1)
        (N, H, W, C) = (2, 7, 7, 2)
        size = 3
        for op in ops:
            x = jt.random([N, H, W, C])
            y = pool(x, size, op)
            mask = jt.random(y.shape)
            loss = (y * mask).sum()
            dx = jt.grad(loss, x)
            jdx = dx.data
            nx = x.data
            nmask = mask.data
            (_, (ndx,)) = ngrad(lambda args: (pool_naive(args[0], size, op) * nmask).sum(), [nx], 1e-06)
            assert np.allclose(jdx, ndx), (op, jdx[0, :, :, 0], ndx[0, :, :, 0])

    def test_fuse_error(self):
        if False:
            i = 10
            return i + 15
        a = jt.array([1, 2, 3, 4])
        b = jt.zeros((3, 3))
        jt.sync_all()
        c = b.reindex_reduce('add', [4, 4], ['@e0(i0)', '@e0(i1)'], extras=[-a])
        c.sync()
        a = jt.zeros((3, 3))
        b = jt.zeros((3, 3))
        jt.sync_all()
        c = b.reindex_reduce('add', [4, 4], ['@e0(i0,i1)', '@e0(i1,i0)'], extras=[-a])
        c.sync()

    def test_error(self):
        if False:
            i = 10
            return i + 15
        jt.random([3]).reindex_reduce('add', [3], ['i0'])
        expect_error(lambda : jt.random([3]).reindex_reduce('add', [3], []))
        expect_error(lambda : jt.random([3]).reindex_reduce('add', [3], ['i0', 'i0']))
        expect_error(lambda : jt.random([3]).reindex_reduce('???', [3], ['i0']))
        expect_error(lambda : jt.random([3]).reindex_reduce('add', [-1], ['i0']))

@unittest.skipIf(not jt.compiler.has_cuda, 'No CUDA found')
class TestReindexReduceOpCuda(TestReindexReduceOp):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        jt.flags.use_cuda = 1

    def tearDown(self):
        if False:
            while True:
                i = 10
        jt.flags.use_cuda = 0
if __name__ == '__main__':
    unittest.main()