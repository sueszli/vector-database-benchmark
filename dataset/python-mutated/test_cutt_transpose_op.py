import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error
from .test_grad import ngrad
from itertools import permutations
from jittor import compile_extern
from .test_log import find_log_with_re
if jt.has_cuda:
    from jittor.compile_extern import cutt_ops
else:
    cutt_ops = None

def gen_data(shape):
    if False:
        for i in range(10):
            print('nop')
    num = np.multiply.reduce(shape)
    a = np.arange(0, num)
    return a.reshape(shape)

class TestCuttTransposeOp(unittest.TestCase):

    @unittest.skipIf(cutt_ops == None, 'Not use cutt, Skip')
    @jt.flag_scope(use_cuda=1)
    def test_with_np(self):
        if False:
            print('Hello World!')

        def check(a):
            if False:
                while True:
                    i = 10
            perms = list(permutations(range(a.ndim))) + [None]
            for perm in perms:
                with jt.log_capture_scope(log_silent=1, log_v=0, log_vprefix='cutt=100') as raw_log:
                    if perm:
                        x = np.transpose(a, perm)
                        y = jt.transpose(a, perm).data
                    else:
                        x = np.transpose(a)
                        y = jt.transpose(a).data
                    self.assertEqual(x.shape, y.shape)
                logs = find_log_with_re(raw_log, '(Run cutt_transpose with key.*)')
                if perm is None:
                    continue
                last = -1
                in_order = True
                for i in range(len(perm)):
                    if a.shape[perm[i]] == 1:
                        continue
                    if last != -1 and last > perm[i]:
                        in_order = False
                        break
                    last = perm[i]
                assert (x == y).all(), f'\n{x}\n{y}\n{perm}\n{a.shape}'
        ia = [gen_data([5, 7]), gen_data([2, 2, 2]), gen_data([2, 3, 4, 5]), gen_data([5, 3]), gen_data([3, 1, 5, 3, 1])]
        for a in ia:
            check(a)

    @unittest.skipIf(cutt_ops == None, 'Not use cutt, Skip')
    @jt.flag_scope(use_cuda=1)
    def test_grad(self):
        if False:
            for i in range(10):
                print('nop')

        def check(a):
            if False:
                for i in range(10):
                    print('nop')
            perms = list(permutations(range(a.ndim))) + [None]
            for perm in perms:
                x = jt.array(a).float()
                if perm:
                    y = jt.transpose(x, perm)
                else:
                    y = jt.transpose(x)
                dx = jt.grad(y * y, x).data
                self.assertEqual(dx.shape, a.shape)
                assert (dx == a * 2).all(), f'\n{dx}\n{a}\n{perm}'
        ia = [gen_data([2, 2, 2]), gen_data([2, 3, 4, 5]), gen_data([5, 3]), gen_data([3, 1, 5, 3, 1])]
        for a in ia:
            check(a)

    @unittest.skipIf(cutt_ops == None, 'Not use cutt, Skip')
    @jt.flag_scope(use_cuda=1)
    def test_matmul_grad(self):
        if False:
            return 10
        np.random.seed(0)
        for i in range(10):
            a = np.random.rand(2, 3).astype('float32')
            b = np.random.rand(3, 4).astype('float32')
            (out, (da, db)) = ngrad(lambda vars: np.matmul(vars[0], vars[1]).sum(), [a, b], 0.1)
            ja = jt.array(a)
            jb = jt.array(b)
            jc = ja.matmul(jb)
            (jda, jdb) = jt.grad(jc, [ja, jb])
            assert (da - jda.data < 1e-05).all(), (da, jda.data, da - jda.data)
            assert (db - jdb.data < 1e-05).all(), db - jdb.data

    @unittest.skipIf(cutt_ops == None, 'Not use cutt, Skip')
    @jt.flag_scope(use_cuda=1)
    def test_matmul_grad(self):
        if False:
            for i in range(10):
                print('nop')
        a = jt.zeros((0, 10))
        b = a.transpose(1, 0)
        c = b.data
        assert c.shape[0] == 10
        assert c.shape[1] == 0
if __name__ == '__main__':
    unittest.main()