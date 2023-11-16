import functools
import queue
import threading
from unittest import skipIf as skipif, SkipTest
import pytest
from pytest import raises as assert_raises
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_TORCHDYNAMO, TestCase
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.random import random
    from numpy.testing import assert_allclose
else:
    import torch._numpy as np
    from torch._numpy.random import random
    from torch._numpy.testing import assert_allclose
skip = functools.partial(skipif, True)
IS_WASM = False

def fft1(x):
    if False:
        return 10
    L = len(x)
    phase = -2j * np.pi * (np.arange(L) / L)
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x * np.exp(phase), axis=1)

class TestFFTShift(TestCase):

    def test_fft_n(self):
        if False:
            i = 10
            return i + 15
        assert_raises((ValueError, RuntimeError), np.fft.fft, [1, 2, 3], 0)

@instantiate_parametrized_tests
class TestFFT1D(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        np.random.seed(123456)

    def test_identity(self):
        if False:
            print('Hello World!')
        maxlen = 512
        x = random(maxlen) + 1j * random(maxlen)
        xr = random(maxlen)
        for i in range(1, maxlen):
            assert_allclose(np.fft.ifft(np.fft.fft(x[0:i])), x[0:i], atol=1e-12)
            assert_allclose(np.fft.irfft(np.fft.rfft(xr[0:i]), i), xr[0:i], atol=1e-12)

    def test_fft(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(1234)
        x = random(30) + 1j * random(30)
        assert_allclose(fft1(x), np.fft.fft(x), atol=3e-05)
        assert_allclose(fft1(x), np.fft.fft(x, norm='backward'), atol=3e-05)
        assert_allclose(fft1(x) / np.sqrt(30), np.fft.fft(x, norm='ortho'), atol=5e-06)
        assert_allclose(fft1(x) / 30.0, np.fft.fft(x, norm='forward'), atol=5e-06)

    @parametrize('norm', (None, 'backward', 'ortho', 'forward'))
    def test_ifft(self, norm):
        if False:
            i = 10
            return i + 15
        x = random(30) + 1j * random(30)
        assert_allclose(x, np.fft.ifft(np.fft.fft(x, norm=norm), norm=norm), atol=1e-06)
        with pytest.raises((ValueError, RuntimeError), match='Invalid number of'):
            np.fft.ifft([], norm=norm)

    def test_fft2(self):
        if False:
            return 10
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(np.fft.fft(np.fft.fft(x, axis=1), axis=0), np.fft.fft2(x), atol=1e-06)
        assert_allclose(np.fft.fft2(x), np.fft.fft2(x, norm='backward'), atol=1e-06)
        assert_allclose(np.fft.fft2(x) / np.sqrt(30 * 20), np.fft.fft2(x, norm='ortho'), atol=1e-06)
        assert_allclose(np.fft.fft2(x) / (30.0 * 20.0), np.fft.fft2(x, norm='forward'), atol=1e-06)

    def test_ifft2(self):
        if False:
            print('Hello World!')
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(np.fft.ifft(np.fft.ifft(x, axis=1), axis=0), np.fft.ifft2(x), atol=1e-06)
        assert_allclose(np.fft.ifft2(x), np.fft.ifft2(x, norm='backward'), atol=1e-06)
        assert_allclose(np.fft.ifft2(x) * np.sqrt(30 * 20), np.fft.ifft2(x, norm='ortho'), atol=1e-06)
        assert_allclose(np.fft.ifft2(x) * (30.0 * 20.0), np.fft.ifft2(x, norm='forward'), atol=1e-06)

    def test_fftn(self):
        if False:
            return 10
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(np.fft.fft(np.fft.fft(np.fft.fft(x, axis=2), axis=1), axis=0), np.fft.fftn(x), atol=1e-06)
        assert_allclose(np.fft.fftn(x), np.fft.fftn(x, norm='backward'), atol=1e-06)
        assert_allclose(np.fft.fftn(x) / np.sqrt(30 * 20 * 10), np.fft.fftn(x, norm='ortho'), atol=1e-06)
        assert_allclose(np.fft.fftn(x) / (30.0 * 20.0 * 10.0), np.fft.fftn(x, norm='forward'), atol=1e-06)

    def test_ifftn(self):
        if False:
            i = 10
            return i + 15
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(np.fft.ifft(np.fft.ifft(np.fft.ifft(x, axis=2), axis=1), axis=0), np.fft.ifftn(x), atol=1e-06)
        assert_allclose(np.fft.ifftn(x), np.fft.ifftn(x, norm='backward'), atol=1e-06)
        assert_allclose(np.fft.ifftn(x) * np.sqrt(30 * 20 * 10), np.fft.ifftn(x, norm='ortho'), atol=1e-06)
        assert_allclose(np.fft.ifftn(x) * (30.0 * 20.0 * 10.0), np.fft.ifftn(x, norm='forward'), atol=1e-06)

    def test_rfft(self):
        if False:
            i = 10
            return i + 15
        x = random(30)
        for n in [x.size, 2 * x.size]:
            for norm in [None, 'backward', 'ortho', 'forward']:
                assert_allclose(np.fft.fft(x, n=n, norm=norm)[:n // 2 + 1], np.fft.rfft(x, n=n, norm=norm), atol=1e-06)
            assert_allclose(np.fft.rfft(x, n=n), np.fft.rfft(x, n=n, norm='backward'), atol=1e-06)
            assert_allclose(np.fft.rfft(x, n=n) / np.sqrt(n), np.fft.rfft(x, n=n, norm='ortho'), atol=1e-06)
            assert_allclose(np.fft.rfft(x, n=n) / n, np.fft.rfft(x, n=n, norm='forward'), atol=1e-06)

    def test_irfft(self):
        if False:
            for i in range(10):
                print('nop')
        x = random(30)
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x)), atol=1e-06)
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x, norm='backward'), norm='backward'), atol=1e-06)
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x, norm='ortho'), norm='ortho'), atol=1e-06)
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x, norm='forward'), norm='forward'), atol=1e-06)

    def test_rfft2(self):
        if False:
            for i in range(10):
                print('nop')
        x = random((30, 20))
        assert_allclose(np.fft.fft2(x)[:, :11], np.fft.rfft2(x), atol=1e-06)
        assert_allclose(np.fft.rfft2(x), np.fft.rfft2(x, norm='backward'), atol=1e-06)
        assert_allclose(np.fft.rfft2(x) / np.sqrt(30 * 20), np.fft.rfft2(x, norm='ortho'), atol=1e-06)
        assert_allclose(np.fft.rfft2(x) / (30.0 * 20.0), np.fft.rfft2(x, norm='forward'), atol=1e-06)

    def test_irfft2(self):
        if False:
            while True:
                i = 10
        x = random((30, 20))
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x)), atol=1e-06)
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x, norm='backward'), norm='backward'), atol=1e-06)
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x, norm='ortho'), norm='ortho'), atol=1e-06)
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x, norm='forward'), norm='forward'), atol=1e-06)

    def test_rfftn(self):
        if False:
            print('Hello World!')
        x = random((30, 20, 10))
        assert_allclose(np.fft.fftn(x)[:, :, :6], np.fft.rfftn(x), atol=1e-06)
        assert_allclose(np.fft.rfftn(x), np.fft.rfftn(x, norm='backward'), atol=1e-06)
        assert_allclose(np.fft.rfftn(x) / np.sqrt(30 * 20 * 10), np.fft.rfftn(x, norm='ortho'), atol=1e-06)
        assert_allclose(np.fft.rfftn(x) / (30.0 * 20.0 * 10.0), np.fft.rfftn(x, norm='forward'), atol=1e-06)

    def test_irfftn(self):
        if False:
            for i in range(10):
                print('nop')
        x = random((30, 20, 10))
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x)), atol=1e-06)
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x, norm='backward'), norm='backward'), atol=1e-06)
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x, norm='ortho'), norm='ortho'), atol=1e-06)
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x, norm='forward'), norm='forward'), atol=1e-06)

    def test_hfft(self):
        if False:
            i = 10
            return i + 15
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, np.flip(x).conj()))
        assert_allclose(np.fft.fft(x), np.fft.hfft(x_herm), atol=1e-06)
        assert_allclose(np.fft.hfft(x_herm), np.fft.hfft(x_herm, norm='backward'), atol=1e-06)
        assert_allclose(np.fft.hfft(x_herm) / np.sqrt(30), np.fft.hfft(x_herm, norm='ortho'), atol=1e-06)
        assert_allclose(np.fft.hfft(x_herm) / 30.0, np.fft.hfft(x_herm, norm='forward'), atol=1e-06)

    def test_ihfft(self):
        if False:
            for i in range(10):
                print('nop')
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, np.flip(x).conj()))
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm)), atol=1e-06)
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm, norm='backward'), norm='backward'), atol=1e-06)
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm, norm='ortho'), norm='ortho'), atol=1e-06)
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm, norm='forward'), norm='forward'), atol=1e-06)

    @parametrize('op', [np.fft.fftn, np.fft.ifftn, np.fft.rfftn, np.fft.irfftn])
    def test_axes(self, op):
        if False:
            for i in range(10):
                print('nop')
        x = random((30, 20, 10))
        axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        for a in axes:
            op_tr = op(np.transpose(x, a))
            tr_op = np.transpose(op(x, axes=a), a)
            assert_allclose(op_tr, tr_op, atol=1e-06)

    def test_all_1d_norm_preserving(self):
        if False:
            i = 10
            return i + 15
        x = random(30)
        x_norm = np.linalg.norm(x)
        n = x.size * 2
        func_pairs = [(np.fft.fft, np.fft.ifft), (np.fft.rfft, np.fft.irfft), (np.fft.ihfft, np.fft.hfft)]
        for (forw, back) in func_pairs:
            for n in [x.size, 2 * x.size]:
                for norm in [None, 'backward', 'ortho', 'forward']:
                    tmp = forw(x, n=n, norm=norm)
                    tmp = back(tmp, n=n, norm=norm)
                    assert_allclose(x_norm, np.linalg.norm(tmp), atol=1e-06)

    @parametrize('dtype', [np.half, np.single, np.double])
    def test_dtypes(self, dtype):
        if False:
            i = 10
            return i + 15
        x = random(30).astype(dtype)
        assert_allclose(np.fft.ifft(np.fft.fft(x)), x, atol=1e-06)
        assert_allclose(np.fft.irfft(np.fft.rfft(x)), x, atol=1e-06)

    @parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
    @parametrize('order', ['F', 'non-contiguous'])
    @parametrize('fft', [np.fft.fft, np.fft.fft2, np.fft.fftn, np.fft.ifft, np.fft.ifft2, np.fft.ifftn])
    def test_fft_with_order(self, dtype, order, fft):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random
        X = rng.rand(8, 7, 13).astype(dtype)
        _tol = float(8.0 * np.sqrt(np.log2(X.size)) * np.finfo(X.dtype).eps)
        if order == 'F':
            raise SkipTest('Fortran order arrays')
            Y = np.asfortranarray(X)
        else:
            Z = np.empty((16, 7, 13), dtype=X.dtype)
            Z[::2] = X
            Y = Z[::2]
            X = Y.copy()
        if fft.__name__.endswith('fft'):
            for axis in range(3):
                X_res = fft(X, axis=axis)
                Y_res = fft(Y, axis=axis)
                assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
        elif fft.__name__.endswith(('fft2', 'fftn')):
            axes = [(0, 1), (1, 2), (0, 2)]
            if fft.__name__.endswith('fftn'):
                axes.extend([(0,), (1,), (2,), None])
            for ax in axes:
                X_res = fft(X, axes=ax)
                Y_res = fft(Y, axes=ax)
                assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
        else:
            raise ValueError()

@skipif(IS_WASM, reason='Cannot start thread')
class TestFFTThreadSafe(TestCase):
    threads = 16
    input_shape = (800, 200)

    def _test_mtsame(self, func, *args):
        if False:
            print('Hello World!')

        def worker(args, q):
            if False:
                print('Hello World!')
            q.put(func(*args))
        q = queue.Queue()
        expected = func(*args)
        t = [threading.Thread(target=worker, args=(args, q)) for i in range(self.threads)]
        [x.start() for x in t]
        [x.join() for x in t]
        for i in range(self.threads):
            assert_allclose(q.get(timeout=5), expected, atol=2e-14)

    def test_fft(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.ones(self.input_shape) * 1 + 0j
        self._test_mtsame(np.fft.fft, a)

    def test_ifft(self):
        if False:
            print('Hello World!')
        a = np.ones(self.input_shape) * 1 + 0j
        self._test_mtsame(np.fft.ifft, a)

    def test_rfft(self):
        if False:
            while True:
                i = 10
        a = np.ones(self.input_shape)
        self._test_mtsame(np.fft.rfft, a)

    def test_irfft(self):
        if False:
            print('Hello World!')
        a = np.ones(self.input_shape) * 1 + 0j
        self._test_mtsame(np.fft.irfft, a)
if __name__ == '__main__':
    run_tests()