"""Tests for FFT via the XLA JIT."""
import itertools
import numpy as np
import scipy.signal as sps
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.signal import signal
from tensorflow.python.platform import googletest
BATCH_DIMS = (3, 5)
RTOL = 0.009
ATOL = 0.0001
RTOL_3D = 0.07
ATOL_3D = 0.0004

def pick_10(x):
    if False:
        for i in range(10):
            print('nop')
    x = list(x)
    np.random.seed(123)
    np.random.shuffle(x)
    return x[:10]

def to_32bit(x):
    if False:
        while True:
            i = 10
    if x.dtype == np.complex128:
        return x.astype(np.complex64)
    if x.dtype == np.float64:
        return x.astype(np.float32)
    return x
POWS_OF_2 = 2 ** np.arange(3, 12)
INNER_DIMS_1D = list(((x,) for x in POWS_OF_2))
POWS_OF_2 = 2 ** np.arange(3, 8)
INNER_DIMS_2D = pick_10(itertools.product(POWS_OF_2, POWS_OF_2))
INNER_DIMS_3D = pick_10(itertools.product(POWS_OF_2, POWS_OF_2, POWS_OF_2))

class FFTTest(xla_test.XLATestCase):

    def _VerifyFftMethod(self, inner_dims, complex_to_input, input_to_expected, tf_method, atol=ATOL, rtol=RTOL):
        if False:
            return 10
        for indims in inner_dims:
            print('nfft =', indims)
            shape = BATCH_DIMS + indims
            data = np.arange(np.prod(shape) * 2) / np.prod(indims)
            np.random.seed(123)
            np.random.shuffle(data)
            data = np.reshape(data.astype(np.float32).view(np.complex64), shape)
            data = to_32bit(complex_to_input(data))
            expected = to_32bit(input_to_expected(data))
            with self.session() as sess:
                with self.test_scope():
                    ph = array_ops.placeholder(dtypes.as_dtype(data.dtype), shape=data.shape)
                    out = tf_method(ph)
                value = sess.run(out, {ph: data})
                self.assertAllClose(expected, value, rtol=rtol, atol=atol)

    def testContribSignalSTFT(self):
        if False:
            for i in range(10):
                print('nop')
        ws = 512
        hs = 128
        dims = (ws * 20,)
        shape = BATCH_DIMS + dims
        data = np.arange(np.prod(shape)) / np.prod(dims)
        np.random.seed(123)
        np.random.shuffle(data)
        data = np.reshape(data.astype(np.float32), shape)
        window = sps.get_window('hann', ws)
        expected = sps.stft(data, nperseg=ws, noverlap=ws - hs, boundary=None, window=window)[2]
        expected = np.swapaxes(expected, -1, -2)
        expected *= window.sum()
        with self.session() as sess:
            with self.test_scope():
                ph = array_ops.placeholder(dtypes.as_dtype(data.dtype), shape=data.shape)
                out = signal.stft(ph, ws, hs)
                grad = gradients_impl.gradients(out, ph, grad_ys=array_ops.ones_like(out))
            (value, _) = sess.run([out, grad], {ph: data})
            self.assertAllClose(expected, value, rtol=RTOL, atol=ATOL)

    def testFFT(self):
        if False:
            print('Hello World!')
        self._VerifyFftMethod(INNER_DIMS_1D, lambda x: x, np.fft.fft, signal.fft)

    def testFFT2D(self):
        if False:
            for i in range(10):
                print('nop')
        self._VerifyFftMethod(INNER_DIMS_2D, lambda x: x, np.fft.fft2, signal.fft2d)

    def testFFT3D(self):
        if False:
            while True:
                i = 10
        self._VerifyFftMethod(INNER_DIMS_3D, lambda x: x, lambda x: np.fft.fftn(x, axes=(-3, -2, -1)), signal.fft3d, ATOL_3D, RTOL_3D)

    def testIFFT(self):
        if False:
            i = 10
            return i + 15
        self._VerifyFftMethod(INNER_DIMS_1D, lambda x: x, np.fft.ifft, signal.ifft)

    def testIFFT2D(self):
        if False:
            while True:
                i = 10
        self._VerifyFftMethod(INNER_DIMS_2D, lambda x: x, np.fft.ifft2, signal.ifft2d)

    def testIFFT3D(self):
        if False:
            return 10
        self._VerifyFftMethod(INNER_DIMS_3D, lambda x: x, lambda x: np.fft.ifftn(x, axes=(-3, -2, -1)), signal.ifft3d, ATOL_3D, RTOL_3D)

    def testRFFT(self):
        if False:
            for i in range(10):
                print('nop')

        def _to_expected(x):
            if False:
                i = 10
                return i + 15
            return np.fft.rfft(x, n=x.shape[-1])

        def _tf_fn(x):
            if False:
                return 10
            return signal.rfft(x, fft_length=[x.shape[-1]])
        self._VerifyFftMethod(INNER_DIMS_1D, np.real, _to_expected, _tf_fn)

    def testRFFT2D(self):
        if False:
            i = 10
            return i + 15

        def _tf_fn(x):
            if False:
                while True:
                    i = 10
            return signal.rfft2d(x, fft_length=[x.shape[-2], x.shape[-1]])
        self._VerifyFftMethod(INNER_DIMS_2D, np.real, lambda x: np.fft.rfft2(x, s=[x.shape[-2], x.shape[-1]]), _tf_fn)

    def testRFFT3D(self):
        if False:
            for i in range(10):
                print('nop')

        def _to_expected(x):
            if False:
                i = 10
                return i + 15
            return np.fft.rfftn(x, axes=(-3, -2, -1), s=[x.shape[-3], x.shape[-2], x.shape[-1]])

        def _tf_fn(x):
            if False:
                i = 10
                return i + 15
            return signal.rfft3d(x, fft_length=[x.shape[-3], x.shape[-2], x.shape[-1]])
        self._VerifyFftMethod(INNER_DIMS_3D, np.real, _to_expected, _tf_fn, ATOL_3D, RTOL_3D)

    def testRFFT3DMismatchedSize(self):
        if False:
            print('Hello World!')

        def _to_expected(x):
            if False:
                i = 10
                return i + 15
            return np.fft.rfftn(x, axes=(-3, -2, -1), s=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

        def _tf_fn(x):
            if False:
                while True:
                    i = 10
            return signal.rfft3d(x, fft_length=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])
        self._VerifyFftMethod(INNER_DIMS_3D, np.real, _to_expected, _tf_fn)

    def testIRFFT(self):
        if False:
            return 10

        def _tf_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return signal.irfft(x, fft_length=[2 * (x.shape[-1] - 1)])
        self._VerifyFftMethod(INNER_DIMS_1D, lambda x: np.fft.rfft(np.real(x), n=x.shape[-1]), lambda x: np.fft.irfft(x, n=2 * (x.shape[-1] - 1)), _tf_fn)

    def testIRFFT2D(self):
        if False:
            print('Hello World!')

        def _tf_fn(x):
            if False:
                while True:
                    i = 10
            return signal.irfft2d(x, fft_length=[x.shape[-2], 2 * (x.shape[-1] - 1)])
        self._VerifyFftMethod(INNER_DIMS_2D, lambda x: np.fft.rfft2(np.real(x), s=[x.shape[-2], x.shape[-1]]), lambda x: np.fft.irfft2(x, s=[x.shape[-2], 2 * (x.shape[-1] - 1)]), _tf_fn)

    def testIRFFT3D(self):
        if False:
            print('Hello World!')

        def _to_input(x):
            if False:
                while True:
                    i = 10
            return np.fft.rfftn(np.real(x), axes=(-3, -2, -1), s=[x.shape[-3], x.shape[-2], x.shape[-1]])

        def _to_expected(x):
            if False:
                i = 10
                return i + 15
            return np.fft.irfftn(x, axes=(-3, -2, -1), s=[x.shape[-3], x.shape[-2], 2 * (x.shape[-1] - 1)])

        def _tf_fn(x):
            if False:
                print('Hello World!')
            return signal.irfft3d(x, fft_length=[x.shape[-3], x.shape[-2], 2 * (x.shape[-1] - 1)])
        self._VerifyFftMethod(INNER_DIMS_3D, _to_input, _to_expected, _tf_fn, ATOL_3D, RTOL_3D)

    def testIRFFT3DMismatchedSize(self):
        if False:
            for i in range(10):
                print('nop')

        def _to_input(x):
            if False:
                return 10
            return np.fft.rfftn(np.real(x), axes=(-3, -2, -1), s=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

        def _to_expected(x):
            if False:
                for i in range(10):
                    print('nop')
            return np.fft.irfftn(x, axes=(-3, -2, -1), s=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

        def _tf_fn(x):
            if False:
                print('Hello World!')
            return signal.irfft3d(x, fft_length=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])
        self._VerifyFftMethod(INNER_DIMS_3D, _to_input, _to_expected, _tf_fn, ATOL_3D, RTOL_3D)
if __name__ == '__main__':
    googletest.main()