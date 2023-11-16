import unittest
import jittor as jt
import numpy as np
from .test_grad import ngrad
from .test_cuda import test_cuda

def check(op, *args):
    if False:
        for i in range(10):
            print('nop')
    x = eval(f'np.{op}(*args)')
    y = eval(f'jt.{op}(*args).data')
    convert = lambda x: x.astype('uint8') if x.dtype == 'bool' else x
    x = convert(x)
    y = convert(y)
    assert x.dtype == y.dtype and x.shape == y.shape, (x.dtype, y.dtype, x.shape, y.shape)
    for (a, b) in zip(x.flatten(), y.flatten()):
        assert str(a)[:5] == str(b)[:5], (a, b)

class TestUnaryOp(unittest.TestCase):

    def test_unary_op(self):
        if False:
            while True:
                i = 10
        assert jt.float64(1).data.dtype == 'float64'
        assert (jt.abs(-1) == 1).data.all()
        assert (abs(-jt.float64(1)) == 1).data.all()
        a = np.array([-1, 2, 3, 0], dtype='int32')
        check('abs', a)
        check('negative', a)
        check('logical_not', a)
        check('bitwise_not', a)
        b = np.array([1.1, 2.2, 3.3, 4.4, -1, 0])
        type = 'float16' if jt.flags.amp_reg & 2 else 'float32'
        check('log', a.astype(type))
        check('exp', a.astype(type))
        check('sqrt', a.astype(type))

    def test_grad(self):
        if False:
            for i in range(10):
                print('nop')
        ops = ['abs', 'negative', 'log', 'exp', 'sqrt', 'sin', 'arcsin', 'sinh', 'arcsinh', 'tan', 'arctan', 'tanh', 'arctanh', 'cos', 'arccos', 'cosh', 'arccosh', 'sigmoid']
        a = np.array([1.1, 2.2, 3.3, 4.4])
        for op in ops:
            if op == 'abs':
                b = np.array(a + [-1])
            elif op == 'arccosh':
                b = np.array(a)
            elif 'sin' in op or 'cos' in op or 'tan' in op:
                b = np.array(a) / 5
            else:
                b = np.array(a)
            func = lambda x: eval(f'np.{op}(x[0]).sum()')
            if op == 'sigmoid':
                func = lambda x: (1 / (1 + np.exp(-x[0]))).sum()
            (x, (da,)) = ngrad(func, [b], 1e-08)
            ja = jt.array(b)
            jb = eval(f'jt.{op}(ja)')
            jda = jt.grad(jb, ja)
            tol = 0.01 if jt.flags.amp_reg & 2 else 1e-06
            assert np.allclose(jda.data, da, atol=tol, rtol=tol), (jda.data, da, op)

    def test_sigmoid(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.arange(-150, 150, 10).astype('float32')
        b = jt.array(a, dtype='float32')
        b1 = b.sigmoid().numpy()
        assert np.isnan(b1).any() == False

    def test_safe_clip(self):
        if False:
            print('Hello World!')
        a = jt.array([-1.0, 0, 0.4, 1, 2, 3])
        b = a.safe_clip(0.1, 0.5)
        assert np.allclose(b.data, [0.1, 0.1, 0.4, 0.5, 0.5, 0.5])
        da = jt.grad(b, a)
        assert (da.data == 1).all()

    def test_erfinv(self):
        if False:
            return 10
        from scipy import special
        y = np.linspace(-1.0, 1.0, num=10)
        x = special.erfinv(y)
        y2 = jt.array(y)
        x2 = jt.erfinv(y2)
        np.testing.assert_allclose(y.data, y2.data)
        y = np.linspace(-0.9, 0.9, num=10)
        x = special.erfinv(y)
        y2 = jt.array(y)
        x2 = jt.erfinv(y2)
        np.testing.assert_allclose(y.data, y2.data)
        d = jt.grad(x2, y2)
        (_, (dn,)) = ngrad(lambda y: special.erfinv(y).sum(), [y], 1e-08)
        tol = 0.001 if jt.flags.amp_reg & 2 else 1e-06
        np.testing.assert_allclose(d.data, dn, atol=tol, rtol=tol)

class TestUnaryOpCuda(TestUnaryOp, test_cuda(2)):
    pass

class TestUnaryOpCpuFp16(TestUnaryOp, test_cuda(2)):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        jt.flags.amp_reg = 2 | 4 | 8 | 16

    def tearDown(self):
        if False:
            while True:
                i = 10
        jt.flags.amp_reg = 0

class TestUnaryOpCudaFp16(TestUnaryOp, test_cuda(2)):

    def setUp(self):
        if False:
            while True:
                i = 10
        jt.flags.amp_reg = 2 | 4 | 8 | 16
        jt.flags.use_cuda = 1

    def tearDown(self):
        if False:
            print('Hello World!')
        jt.flags.amp_reg = 0
        jt.flags.use_cuda = 0
if __name__ == '__main__':
    unittest.main()