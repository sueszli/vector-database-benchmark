"""Tests for fft operations."""
import itertools
import unittest
from absl.testing import parameterized
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test
VALID_FFT_RANKS = (1, 2, 3)

class BaseFFTOpsTest(test.TestCase):

    def _Compare_fftn(self, x, fft_length=None, axes=None, norm=None, use_placeholder=False, rtol=0.0001, atol=0.0001):
        if False:
            while True:
                i = 10
        self._CompareForward_fftn(x, fft_length, axes, norm, use_placeholder, rtol, atol)
        self._CompareBackward_fftn(x, fft_length, axes, norm, use_placeholder, rtol, atol)

    def _CompareForward_fftn(self, x, fft_length=None, axes=None, norm=None, use_placeholder=False, rtol=0.0001, atol=0.0001):
        if False:
            while True:
                i = 10
        x_np = self._np_fftn(x, fft_length, axes, norm)
        if use_placeholder:
            x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
            x_tf = self._tf_fftn(x_ph, fft_length, axes, norm, feed_dict={x_ph: x})
        else:
            x_tf = self._tf_fftn(x, fft_length, axes, norm)
        self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

    def _CompareBackward_fftn(self, x, fft_length=None, axes=None, norm=None, use_placeholder=False, rtol=0.0001, atol=0.0001):
        if False:
            while True:
                i = 10
        x_np = self._np_ifftn(x, fft_length, axes, norm)
        if use_placeholder:
            x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
            x_tf = self._tf_ifftn(x_ph, fft_length, axes, norm, feed_dict={x_ph: x})
        else:
            x_tf = self._tf_ifftn(x, fft_length, axes, norm)
        self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

    def _compare(self, x, rank, fft_length=None, use_placeholder=False, rtol=0.0001, atol=0.0001):
        if False:
            while True:
                i = 10
        self._compare_forward(x, rank, fft_length, use_placeholder, rtol, atol)
        self._compare_backward(x, rank, fft_length, use_placeholder, rtol, atol)

    def _compare_forward(self, x, rank, fft_length=None, use_placeholder=False, rtol=0.0001, atol=0.0001):
        if False:
            while True:
                i = 10
        x_np = self._np_fft(x, rank, fft_length)
        if use_placeholder:
            x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
            x_tf = self._tf_fft(x_ph, rank, fft_length, feed_dict={x_ph: x})
        else:
            x_tf = self._tf_fft(x, rank, fft_length)
        self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

    def _compare_backward(self, x, rank, fft_length=None, use_placeholder=False, rtol=0.0001, atol=0.0001):
        if False:
            i = 10
            return i + 15
        x_np = self._np_ifft(x, rank, fft_length)
        if use_placeholder:
            x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
            x_tf = self._tf_ifft(x_ph, rank, fft_length, feed_dict={x_ph: x})
        else:
            x_tf = self._tf_ifft(x, rank, fft_length)
        self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

    def _check_memory_fail(self, x, rank):
        if False:
            while True:
                i = 10
        config = config_pb2.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.01
        with self.cached_session(config=config, force_gpu=True):
            self._tf_fft(x, rank, fft_length=None)

    def _check_grad_complex(self, func, x, y, result_is_complex=True, rtol=0.01, atol=0.01):
        if False:
            return 10
        with self.cached_session():

            def f(inx, iny):
                if False:
                    i = 10
                    return i + 15
                inx.set_shape(x.shape)
                iny.set_shape(y.shape)
                z = func(math_ops.complex(inx, iny))
                loss = math_ops.reduce_sum(math_ops.real(z * math_ops.conj(z)))
                return loss
            ((x_jacob_t, y_jacob_t), (x_jacob_n, y_jacob_n)) = gradient_checker_v2.compute_gradient(f, [x, y], delta=0.01)
        self.assertAllClose(x_jacob_t, x_jacob_n, rtol=rtol, atol=atol)
        self.assertAllClose(y_jacob_t, y_jacob_n, rtol=rtol, atol=atol)

    def _check_grad_real(self, func, x, rtol=0.01, atol=0.01):
        if False:
            while True:
                i = 10

        def f(inx):
            if False:
                i = 10
                return i + 15
            inx.set_shape(x.shape)
            z = func(inx)
            loss = math_ops.reduce_sum(math_ops.real(z * math_ops.conj(z)))
            return loss
        ((x_jacob_t,), (x_jacob_n,)) = gradient_checker_v2.compute_gradient(f, [x], delta=0.01)
        self.assertAllClose(x_jacob_t, x_jacob_n, rtol=rtol, atol=atol)

@test_util.run_all_in_graph_and_eager_modes
class FFTOpsTest(BaseFFTOpsTest, parameterized.TestCase):

    def _tf_fft(self, x, rank, fft_length=None, feed_dict=None):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            return sess.run(self._tf_fft_for_rank(rank)(x), feed_dict=feed_dict)

    def _tf_ifft(self, x, rank, fft_length=None, feed_dict=None):
        if False:
            return 10
        with self.cached_session() as sess:
            return sess.run(self._tf_ifft_for_rank(rank)(x), feed_dict=feed_dict)

    def _np_fft(self, x, rank, fft_length=None):
        if False:
            i = 10
            return i + 15
        if rank == 1:
            return np.fft.fft2(x, s=fft_length, axes=(-1,))
        elif rank == 2:
            return np.fft.fft2(x, s=fft_length, axes=(-2, -1))
        elif rank == 3:
            return np.fft.fft2(x, s=fft_length, axes=(-3, -2, -1))
        else:
            raise ValueError('invalid rank')

    def _np_ifft(self, x, rank, fft_length=None):
        if False:
            return 10
        if rank == 1:
            return np.fft.ifft2(x, s=fft_length, axes=(-1,))
        elif rank == 2:
            return np.fft.ifft2(x, s=fft_length, axes=(-2, -1))
        elif rank == 3:
            return np.fft.ifft2(x, s=fft_length, axes=(-3, -2, -1))
        else:
            raise ValueError('invalid rank')

    def _tf_fftn(self, x, fft_length=None, axes=None, norm=None, feed_dict=None):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            return sess.run(fft_ops.fftnd(x, fft_length=fft_length, axes=axes, norm=norm), feed_dict=feed_dict)

    def _tf_ifftn(self, x, fft_length=None, axes=None, norm=None, feed_dict=None):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            return sess.run(fft_ops.ifftnd(x, fft_length=fft_length, axes=axes, norm=norm), feed_dict=feed_dict)

    def _np_fftn(self, x, fft_length=None, axes=None, norm=None):
        if False:
            print('Hello World!')
        return np.fft.fftn(x, s=fft_length, axes=axes, norm=norm)

    def _np_ifftn(self, x, fft_length=None, axes=None, norm=None):
        if False:
            i = 10
            return i + 15
        return np.fft.ifftn(x, s=fft_length, axes=axes, norm=norm)

    def _tf_fft_for_rank(self, rank):
        if False:
            print('Hello World!')
        if rank == 1:
            return fft_ops.fft
        elif rank == 2:
            return fft_ops.fft2d
        elif rank == 3:
            return fft_ops.fft3d
        else:
            raise ValueError('invalid rank')

    def _tf_ifft_for_rank(self, rank):
        if False:
            print('Hello World!')
        if rank == 1:
            return fft_ops.ifft
        elif rank == 2:
            return fft_ops.ifft2d
        elif rank == 3:
            return fft_ops.ifft3d
        else:
            raise ValueError('invalid rank')

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (np.complex64, np.complex128)))
    def test_empty(self, rank, extra_dims, np_type):
        if False:
            for i in range(10):
                print('nop')
        dims = rank + extra_dims
        x = np.zeros((0,) * dims).astype(np_type)
        self.assertEqual(x.shape, self._tf_fft(x, rank).shape)
        self.assertEqual(x.shape, self._tf_ifft(x, rank).shape)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (np.complex64, np.complex128)))
    def test_basic(self, rank, extra_dims, np_type):
        if False:
            print('Hello World!')
        dims = rank + extra_dims
        tol = 0.0001 if np_type == np.complex64 else 1e-08
        self._compare(np.mod(np.arange(np.power(4, dims)), 10).reshape((4,) * dims).astype(np_type), rank, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(range(3, 5), (np.complex64, np.complex128)))
    @test_util.run_gpu_only
    def testBasic_fftn(self, dims, np_type):
        if False:
            i = 10
            return i + 15
        fft_length = (4,)
        axes = (-1,)
        tol = 0.0001 if np_type == np.complex64 else 1e-08
        self._Compare_fftn(np.mod(np.arange(np.power(4, dims)), 10).reshape((4,) * dims).astype(np_type), fft_length=fft_length, axes=axes, rtol=tol)

    @parameterized.parameters(itertools.product(range(1, 5), (np.complex64, np.complex128)))
    @test_util.run_gpu_only
    def testFftLength_fftn(self, dims, np_type):
        if False:
            while True:
                i = 10
        tol = 0.0001 if np_type == np.complex64 else 1e-08
        if dims == 1:
            fft_length = (4,)
            axes = (0,)
        elif dims == 2:
            fft_length = (2, 2)
            axes = (0, 1)
        else:
            fft_length = (6, 4, 6)
            axes = (-3, -2, -1)
        self._Compare_fftn(np.mod(np.arange(np.power(4, dims)), 10).reshape((4,) * dims).astype(np_type), fft_length=fft_length, axes=axes, rtol=tol)

    @parameterized.parameters(itertools.product(range(1, 4), (np.complex64, np.complex128)))
    @test_util.run_gpu_only
    def testAxes_fftn(self, dims, np_type):
        if False:
            return 10
        tol = 0.0001 if np_type == np.complex64 else 1e-08
        if dims == 1:
            fft_length = (4,)
            axes = (-1,)
        elif dims == 2:
            fft_length = (4, 4)
            axes = (0, 1)
        else:
            fft_length = None
            axes = None
        self._Compare_fftn(np.mod(np.arange(np.power(4, dims)), 10).reshape((4,) * dims).astype(np_type), fft_length=fft_length, axes=axes, rtol=tol)

    @test_util.run_gpu_only
    def testAxesError_fftn(self):
        if False:
            print('Hello World!')
        with self.assertRaisesWithPredicateMatch(ValueError, 'Shape .* must have rank at least {}.*'.format(2)):
            with self.cached_session():
                self.evaluate(self._tf_fftn(np.zeros((8,)), axes=(1, 0)))
        with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, 'The last axis to perform transform on must be -1'):
            with self.cached_session():
                self.evaluate(self._tf_fftn(np.zeros((8, 8, 8)), axes=(1,)))
        with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, 'axes must be successive and ascending.'):
            with self.cached_session():
                self.evaluate(self._tf_fftn(np.zeros((8, 8, 8)), axes=(1, 0, 2)))
        with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, 'axes must be successive and ascending.'):
            with self.cached_session():
                self.evaluate(self._tf_fftn(np.zeros((8, 8, 8, 8)), axes=(0, 2, -1)))

    @parameterized.parameters(itertools.product(('backward', 'ortho', 'forward'), (np.complex64, np.complex128)))
    @test_util.run_gpu_only
    def testNorm_fftn(self, norm, np_type):
        if False:
            while True:
                i = 10
        tol = 0.0001 if np_type == np.complex64 else 1e-08
        self._Compare_fftn(np.mod(np.arange(np.power(4, 4)), 10).reshape((4,) * 4).astype(np_type), fft_length=(4, 4), axes=(-2, -1), norm=norm, rtol=tol)

    @parameterized.parameters(itertools.product((1,), range(3), (np.complex64, np.complex128)))
    def test_large_batch(self, rank, extra_dims, np_type):
        if False:
            return 10
        dims = rank + extra_dims
        tol = 0.0001 if np_type == np.complex64 else 5e-05
        self._compare(np.mod(np.arange(np.power(128, dims)), 10).reshape((128,) * dims).astype(np_type), rank, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (np.complex64, np.complex128)))
    def test_placeholder(self, rank, extra_dims, np_type):
        if False:
            return 10
        if context.executing_eagerly():
            return
        tol = 0.0001 if np_type == np.complex64 else 1e-08
        dims = rank + extra_dims
        self._compare(np.mod(np.arange(np.power(4, dims)), 10).reshape((4,) * dims).astype(np_type), rank, use_placeholder=True, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (np.complex64, np.complex128)))
    def test_random(self, rank, extra_dims, np_type):
        if False:
            for i in range(10):
                print('nop')
        tol = 0.0001 if np_type == np.complex64 else 5e-06
        dims = rank + extra_dims

        def gen(shape):
            if False:
                for i in range(10):
                    print('nop')
            n = np.prod(shape)
            re = np.random.uniform(size=n)
            im = np.random.uniform(size=n)
            return (re + im * 1j).reshape(shape)
        self._compare(gen((4,) * dims).astype(np_type), rank, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, [128, 256, 512, 1024, 127, 255, 511, 1023], (np.complex64, np.complex128)))
    def test_random_1d(self, rank, dim, np_type):
        if False:
            for i in range(10):
                print('nop')
        has_gpu = test.is_gpu_available(cuda_only=True)
        tol = {(np.complex64, True): 0.0001, (np.complex64, False): 0.01, (np.complex128, True): 0.0001, (np.complex128, False): 0.01}[np_type, has_gpu]

        def gen(shape):
            if False:
                print('Hello World!')
            n = np.prod(shape)
            re = np.random.uniform(size=n)
            im = np.random.uniform(size=n)
            return (re + im * 1j).reshape(shape)
        self._compare(gen((dim,)).astype(np_type), 1, rtol=tol, atol=tol)

    def test_error(self):
        if False:
            print('Hello World!')
        if context.executing_eagerly():
            return
        for rank in VALID_FFT_RANKS:
            for dims in range(0, rank):
                x = np.zeros((1,) * dims).astype(np.complex64)
                with self.assertRaisesWithPredicateMatch(ValueError, 'Shape must be .*rank {}.*'.format(rank)):
                    self._tf_fft(x, rank)
                with self.assertRaisesWithPredicateMatch(ValueError, 'Shape must be .*rank {}.*'.format(rank)):
                    self._tf_ifft(x, rank)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(2), (np.float32, np.float64)))
    def test_grad_simple(self, rank, extra_dims, np_type):
        if False:
            for i in range(10):
                print('nop')
        tol = 0.0001 if np_type == np.float32 else 1e-10
        dims = rank + extra_dims
        re = np.ones(shape=(4,) * dims, dtype=np_type) / 10.0
        im = np.zeros(shape=(4,) * dims, dtype=np_type)
        self._check_grad_complex(self._tf_fft_for_rank(rank), re, im, rtol=tol, atol=tol)
        self._check_grad_complex(self._tf_ifft_for_rank(rank), re, im, rtol=tol, atol=tol)

    @unittest.skip('16.86% flaky')
    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(2), (np.float32, np.float64)))
    def test_grad_random(self, rank, extra_dims, np_type):
        if False:
            while True:
                i = 10
        dims = rank + extra_dims
        tol = 0.01 if np_type == np.float32 else 1e-10
        re = np.random.rand(*(3,) * dims).astype(np_type) * 2 - 1
        im = np.random.rand(*(3,) * dims).astype(np_type) * 2 - 1
        self._check_grad_complex(self._tf_fft_for_rank(rank), re, im, rtol=tol, atol=tol)
        self._check_grad_complex(self._tf_ifft_for_rank(rank), re, im, rtol=tol, atol=tol)

@test_util.run_all_in_graph_and_eager_modes
class RFFTOpsTest(BaseFFTOpsTest, parameterized.TestCase):

    def _tf_fft(self, x, rank, fft_length=None, feed_dict=None):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            return sess.run(self._tf_fft_for_rank(rank)(x, fft_length), feed_dict=feed_dict)

    def _tf_ifft(self, x, rank, fft_length=None, feed_dict=None):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            return sess.run(self._tf_ifft_for_rank(rank)(x, fft_length), feed_dict=feed_dict)

    def _tf_fftn(self, x, fft_length=None, axes=None, norm=None, feed_dict=None):
        if False:
            return 10
        with self.cached_session() as sess:
            return sess.run(fft_ops.rfftnd(x, fft_length=fft_length, axes=axes, norm=norm), feed_dict=feed_dict)

    def _tf_ifftn(self, x, fft_length=None, axes=None, norm=None, feed_dict=None):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            return sess.run(fft_ops.irfftnd(x, fft_length=fft_length, axes=axes, norm=norm), feed_dict=feed_dict)

    def _np_fftn(self, x, fft_length=None, axes=None, norm=None):
        if False:
            for i in range(10):
                print('nop')
        return np.fft.rfftn(x, s=fft_length, axes=axes, norm=norm)

    def _np_ifftn(self, x, fft_length=None, axes=None, norm=None):
        if False:
            i = 10
            return i + 15
        return np.fft.irfftn(x, s=fft_length, axes=axes, norm=norm)

    def _np_fft(self, x, rank, fft_length=None):
        if False:
            while True:
                i = 10
        if rank == 1:
            return np.fft.rfft2(x, s=fft_length, axes=(-1,))
        elif rank == 2:
            return np.fft.rfft2(x, s=fft_length, axes=(-2, -1))
        elif rank == 3:
            return np.fft.rfft2(x, s=fft_length, axes=(-3, -2, -1))
        else:
            raise ValueError('invalid rank')

    def _np_ifft(self, x, rank, fft_length=None):
        if False:
            for i in range(10):
                print('nop')
        if rank == 1:
            return np.fft.irfft2(x, s=fft_length, axes=(-1,))
        elif rank == 2:
            return np.fft.irfft2(x, s=fft_length, axes=(-2, -1))
        elif rank == 3:
            return np.fft.irfft2(x, s=fft_length, axes=(-3, -2, -1))
        else:
            raise ValueError('invalid rank')

    def _tf_fft_for_rank(self, rank):
        if False:
            return 10
        if rank == 1:
            return fft_ops.rfft
        elif rank == 2:
            return fft_ops.rfft2d
        elif rank == 3:
            return fft_ops.rfft3d
        else:
            raise ValueError('invalid rank')

    def _tf_ifft_for_rank(self, rank):
        if False:
            return 10
        if rank == 1:
            return fft_ops.irfft
        elif rank == 2:
            return fft_ops.irfft2d
        elif rank == 3:
            return fft_ops.irfft3d
        else:
            raise ValueError('invalid rank')

    def _generate_valid_irfft_input(self, c2r, np_ctype, r2c, np_rtype, rank, fft_length):
        if False:
            return 10
        if test.is_built_with_rocm():
            return self._np_fft(r2c.astype(np_rtype), rank, fft_length)
        else:
            return c2r.astype(np_ctype)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (np.float32, np.float64)))
    def test_empty(self, rank, extra_dims, np_rtype):
        if False:
            while True:
                i = 10
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        dims = rank + extra_dims
        x = np.zeros((0,) * dims).astype(np_rtype)
        self.assertEqual(x.shape, self._tf_fft(x, rank).shape)
        x = np.zeros((0,) * dims).astype(np_ctype)
        self.assertEqual(x.shape, self._tf_ifft(x, rank).shape)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
    def test_basic(self, rank, extra_dims, size, np_rtype):
        if False:
            print('Hello World!')
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_rtype == np.float32 else 5e-05
        dims = rank + extra_dims
        inner_dim = size // 2 + 1
        r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
        fft_length = (size,) * rank
        self._compare_forward(r2c.astype(np_rtype), rank, fft_length, rtol=tol, atol=tol)
        c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape((size,) * (dims - 1) + (inner_dim,))
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank, fft_length)
        self._compare_backward(c2r, rank, fft_length, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(range(3, 5), (5, 6), (np.float32, np.float64)))
    @test_util.run_gpu_only
    def testBasic_rfftn(self, dims, size, np_rtype):
        if False:
            print('Hello World!')
        fft_length = (size, size)
        axes = (-2, -1)
        inner_dim = size // 2 + 1
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_ctype == np.complex64 else 1e-08
        r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
        self._CompareForward_fftn(r2c.astype(np_rtype), fft_length=fft_length, axes=axes, rtol=tol)
        c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape((size,) * (dims - 1) + (inner_dim,))
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, 2, fft_length)
        self._CompareBackward_fftn(c2r, fft_length, axes, rtol=tol)

    @parameterized.parameters(itertools.product(range(1, 5), (5, 6), (np.float32, np.float64)))
    @test_util.run_gpu_only
    def testFftLength_rfftn(self, dims, size, np_rtype):
        if False:
            return 10
        inner_dim = size // 2 + 1
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_ctype == np.complex64 else 1e-08
        r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
        if dims == 1:
            fft_length = (size,)
            axes = (-1,)
        elif dims == 2:
            fft_length = (size // 2, size // 2)
            axes = (-2, -1)
        else:
            fft_length = (size * 2, size, size * 2)
            axes = (-3, -2, -1)
        self._CompareBackward_fftn(r2c.astype(np_rtype), fft_length=fft_length, axes=axes, rtol=tol)
        c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape((size,) * (dims - 1) + (inner_dim,))
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, 2, fft_length)
        self._CompareForward_fftn(c2r, fft_length, axes, rtol=tol)

    @parameterized.parameters(itertools.product(range(1, 4), (5, 6), (np.float32, np.float64)))
    @test_util.run_gpu_only
    def testAxes_rfftn(self, dims, size, np_rtype):
        if False:
            for i in range(10):
                print('nop')
        inner_dim = size // 2 + 1
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_ctype == np.complex64 else 1e-08
        r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
        if dims == 1:
            fft_length = (size,)
            axes = (-1,)
        elif dims == 2:
            fft_length = (size, size)
            axes = (0, 1)
        else:
            fft_length = None
            axes = None
        self._CompareForward_fftn(r2c.astype(np_rtype), fft_length=fft_length, axes=axes, rtol=tol)
        c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape((size,) * (dims - 1) + (inner_dim,))
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, 2, fft_length)
        self._CompareBackward_fftn(c2r, fft_length, axes, rtol=tol)

    @parameterized.parameters(itertools.product(('backward', 'ortho', 'forward'), (np.float32, np.float64)))
    @test_util.run_gpu_only
    def testNorm_rfftn(self, norm, np_rtype):
        if False:
            while True:
                i = 10
        inner_dim = 3
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_ctype == np.complex64 else 1e-08
        r2c = np.mod(np.arange(np.power(5, 4)), 10).reshape((5,) * 4)
        fft_length = (5, 5)
        axes = (-2, -1)
        self._CompareForward_fftn(r2c.astype(np_rtype), fft_length=fft_length, axes=axes, norm=norm, rtol=tol)
        c2r = np.mod(np.arange(np.power(5, 4 - 1) * inner_dim), 10).reshape((5,) * (4 - 1) + (inner_dim,))
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, 2, fft_length)
        self._CompareBackward_fftn(c2r, fft_length, axes, norm=norm, rtol=tol)

    @parameterized.parameters(itertools.product((1,), range(3), (64, 128), (np.float32, np.float64)))
    def test_large_batch(self, rank, extra_dims, size, np_rtype):
        if False:
            print('Hello World!')
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_rtype == np.float32 else 1e-05
        dims = rank + extra_dims
        inner_dim = size // 2 + 1
        r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
        fft_length = (size,) * rank
        self._compare_forward(r2c.astype(np_rtype), rank, fft_length, rtol=tol, atol=tol)
        c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape((size,) * (dims - 1) + (inner_dim,))
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank, fft_length)
        self._compare_backward(c2r, rank, fft_length, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
    def test_placeholder(self, rank, extra_dims, size, np_rtype):
        if False:
            for i in range(10):
                print('nop')
        if context.executing_eagerly():
            return
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_rtype == np.float32 else 1e-08
        dims = rank + extra_dims
        inner_dim = size // 2 + 1
        r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
        fft_length = (size,) * rank
        self._compare_forward(r2c.astype(np_rtype), rank, fft_length, use_placeholder=True, rtol=tol, atol=tol)
        c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape((size,) * (dims - 1) + (inner_dim,))
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank, fft_length)
        self._compare_backward(c2r, rank, fft_length, use_placeholder=True, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
    def test_fft_lenth_truncate(self, rank, extra_dims, size, np_rtype):
        if False:
            i = 10
            return i + 15
        'Test truncation (FFT size < dimensions).'
        if test.is_built_with_rocm() and rank == 3:
            self.skipTest('Test fails on ROCm...fix me')
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_rtype == np.float32 else 8e-05
        dims = rank + extra_dims
        inner_dim = size // 2 + 1
        r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
        c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape((size,) * (dims - 1) + (inner_dim,))
        fft_length = (size - 2,) * rank
        self._compare_forward(r2c.astype(np_rtype), rank, fft_length, rtol=tol, atol=tol)
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank, fft_length)
        self._compare_backward(c2r, rank, fft_length, rtol=tol, atol=tol)
        if not context.executing_eagerly():
            self._compare_forward(r2c.astype(np_rtype), rank, fft_length, use_placeholder=True, rtol=tol, atol=tol)
            self._compare_backward(c2r, rank, fft_length, use_placeholder=True, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
    def test_fft_lenth_pad(self, rank, extra_dims, size, np_rtype):
        if False:
            while True:
                i = 10
        'Test padding (FFT size > dimensions).'
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_rtype == np.float32 else 8e-05
        dims = rank + extra_dims
        inner_dim = size // 2 + 1
        r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape((size,) * dims)
        c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim), 10).reshape((size,) * (dims - 1) + (inner_dim,))
        fft_length = (size + 2,) * rank
        self._compare_forward(r2c.astype(np_rtype), rank, fft_length, rtol=tol, atol=tol)
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank, fft_length)
        self._compare_backward(c2r.astype(np_ctype), rank, fft_length, rtol=tol, atol=tol)
        if not context.executing_eagerly():
            self._compare_forward(r2c.astype(np_rtype), rank, fft_length, use_placeholder=True, rtol=tol, atol=tol)
            self._compare_backward(c2r.astype(np_ctype), rank, fft_length, use_placeholder=True, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
    def test_random(self, rank, extra_dims, size, np_rtype):
        if False:
            for i in range(10):
                print('nop')

        def gen_real(shape):
            if False:
                return 10
            n = np.prod(shape)
            re = np.random.uniform(size=n)
            ret = re.reshape(shape)
            return ret

        def gen_complex(shape):
            if False:
                for i in range(10):
                    print('nop')
            n = np.prod(shape)
            re = np.random.uniform(size=n)
            im = np.random.uniform(size=n)
            ret = (re + im * 1j).reshape(shape)
            return ret
        np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
        tol = 0.0001 if np_rtype == np.float32 else 1e-05
        dims = rank + extra_dims
        r2c = gen_real((size,) * dims)
        inner_dim = size // 2 + 1
        fft_length = (size,) * rank
        self._compare_forward(r2c.astype(np_rtype), rank, fft_length, rtol=tol, atol=tol)
        complex_dims = (size,) * (dims - 1) + (inner_dim,)
        c2r = gen_complex(complex_dims)
        c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank, fft_length)
        self._compare_backward(c2r, rank, fft_length, rtol=tol, atol=tol)

    def test_error(self):
        if False:
            while True:
                i = 10
        if context.executing_eagerly():
            return
        for rank in VALID_FFT_RANKS:
            for dims in range(0, rank):
                x = np.zeros((1,) * dims).astype(np.complex64)
                with self.assertRaisesWithPredicateMatch(ValueError, 'Shape .* must have rank at least {}'.format(rank)):
                    self._tf_fft(x, rank)
                with self.assertRaisesWithPredicateMatch(ValueError, 'Shape .* must have rank at least {}'.format(rank)):
                    self._tf_ifft(x, rank)
            for dims in range(rank, rank + 2):
                x = np.zeros((1,) * rank)
                fft_length = np.zeros((1, 1)).astype(np.int32)
                with self.assertRaisesWithPredicateMatch(ValueError, 'Shape .* must have rank 1'):
                    self._tf_fft(x, rank, fft_length)
                with self.assertRaisesWithPredicateMatch(ValueError, 'Shape .* must have rank 1'):
                    self._tf_ifft(x, rank, fft_length)
                fft_length = np.zeros((rank + 1,)).astype(np.int32)
                with self.assertRaisesWithPredicateMatch(ValueError, 'Dimension must be .*but is {}.*'.format(rank + 1)):
                    self._tf_fft(x, rank, fft_length)
                with self.assertRaisesWithPredicateMatch(ValueError, 'Dimension must be .*but is {}.*'.format(rank + 1)):
                    self._tf_ifft(x, rank, fft_length)
            rffts_for_rank = {1: [gen_spectral_ops.rfft, gen_spectral_ops.irfft], 2: [gen_spectral_ops.rfft2d, gen_spectral_ops.irfft2d], 3: [gen_spectral_ops.rfft3d, gen_spectral_ops.irfft3d]}
            (rfft_fn, irfft_fn) = rffts_for_rank[rank]
            with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, 'Input dimension .* must have length of at least 6 but got: 5'):
                x = np.zeros((5,) * rank).astype(np.float32)
                fft_length = [6] * rank
                with self.cached_session():
                    self.evaluate(rfft_fn(x, fft_length))
            with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, 'Input dimension .* must have length of at least .* but got: 3'):
                x = np.zeros((3,) * rank).astype(np.complex64)
                fft_length = [6] * rank
                with self.cached_session():
                    self.evaluate(irfft_fn(x, fft_length))

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(2), (5, 6), (np.float32, np.float64)))
    def test_grad_simple(self, rank, extra_dims, size, np_rtype):
        if False:
            for i in range(10):
                print('nop')
        if rank == 3:
            return
        dims = rank + extra_dims
        tol = 0.001 if np_rtype == np.float32 else 1e-10
        re = np.ones(shape=(size,) * dims, dtype=np_rtype)
        im = -np.ones(shape=(size,) * dims, dtype=np_rtype)
        self._check_grad_real(self._tf_fft_for_rank(rank), re, rtol=tol, atol=tol)
        if test.is_built_with_rocm():
            return
        self._check_grad_complex(self._tf_ifft_for_rank(rank), re, im, result_is_complex=False, rtol=tol, atol=tol)

    @parameterized.parameters(itertools.product(VALID_FFT_RANKS, range(2), (5, 6), (np.float32, np.float64)))
    def test_grad_random(self, rank, extra_dims, size, np_rtype):
        if False:
            i = 10
            return i + 15
        if rank == 3:
            return
        dims = rank + extra_dims
        tol = 0.01 if np_rtype == np.float32 else 1e-10
        re = np.random.rand(*(size,) * dims).astype(np_rtype) * 2 - 1
        im = np.random.rand(*(size,) * dims).astype(np_rtype) * 2 - 1
        self._check_grad_real(self._tf_fft_for_rank(rank), re, rtol=tol, atol=tol)
        if test.is_built_with_rocm():
            return
        self._check_grad_complex(self._tf_ifft_for_rank(rank), re, im, result_is_complex=False, rtol=tol, atol=tol)

    def test_invalid_args(self):
        if False:
            return 10
        a = np.empty([6, 0])
        b = np.array([1, -1])
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), '(.*must be greater or equal to.*)|(must >= 0)'):
            with self.session():
                v = fft_ops.rfft2d(input_tensor=a, fft_length=b)
                self.evaluate(v)

@test_util.run_all_in_graph_and_eager_modes
class FFTShiftTest(test.TestCase, parameterized.TestCase):

    def test_definition(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
            y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            self.assertAllEqual(fft_ops.fftshift(x), y)
            self.assertAllEqual(fft_ops.ifftshift(y), x)
            x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
            y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
            self.assertAllEqual(fft_ops.fftshift(x), y)
            self.assertAllEqual(fft_ops.ifftshift(y), x)

    def test_axes_keyword(self):
        if False:
            while True:
                i = 10
        with self.session():
            freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
            shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
            self.assertAllEqual(fft_ops.fftshift(freqs, axes=(0, 1)), shifted)
            self.assertAllEqual(fft_ops.fftshift(freqs, axes=0), fft_ops.fftshift(freqs, axes=(0,)))
            self.assertAllEqual(fft_ops.ifftshift(shifted, axes=(0, 1)), freqs)
            self.assertAllEqual(fft_ops.ifftshift(shifted, axes=0), fft_ops.ifftshift(shifted, axes=(0,)))
            self.assertAllEqual(fft_ops.fftshift(freqs), shifted)
            self.assertAllEqual(fft_ops.ifftshift(shifted), freqs)

    def test_numpy_compatibility(self):
        if False:
            print('Hello World!')
        with self.session():
            x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
            y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            self.assertAllEqual(fft_ops.fftshift(x), np.fft.fftshift(x))
            self.assertAllEqual(fft_ops.ifftshift(y), np.fft.ifftshift(y))
            x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
            y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
            self.assertAllEqual(fft_ops.fftshift(x), np.fft.fftshift(x))
            self.assertAllEqual(fft_ops.ifftshift(y), np.fft.ifftshift(y))
            freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
            shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
            self.assertAllEqual(fft_ops.fftshift(freqs, axes=(0, 1)), np.fft.fftshift(freqs, axes=(0, 1)))
            self.assertAllEqual(fft_ops.ifftshift(shifted, axes=(0, 1)), np.fft.ifftshift(shifted, axes=(0, 1)))

    @parameterized.parameters(None, 1, ([1, 2],))
    def test_placeholder(self, axes):
        if False:
            while True:
                i = 10
        if context.executing_eagerly():
            return
        x = array_ops.placeholder(shape=[None, None, None], dtype='float32')
        y_fftshift = fft_ops.fftshift(x, axes=axes)
        y_ifftshift = fft_ops.ifftshift(x, axes=axes)
        x_np = np.random.rand(16, 256, 256)
        with self.session() as sess:
            (y_fftshift_res, y_ifftshift_res) = sess.run([y_fftshift, y_ifftshift], feed_dict={x: x_np})
        self.assertAllClose(y_fftshift_res, np.fft.fftshift(x_np, axes=axes))
        self.assertAllClose(y_ifftshift_res, np.fft.ifftshift(x_np, axes=axes))

    def test_negative_axes(self):
        if False:
            i = 10
            return i + 15
        with self.session():
            freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
            shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
            self.assertAllEqual(fft_ops.fftshift(freqs, axes=(0, -1)), shifted)
            self.assertAllEqual(fft_ops.ifftshift(shifted, axes=(0, -1)), freqs)
            self.assertAllEqual(fft_ops.fftshift(freqs, axes=-1), fft_ops.fftshift(freqs, axes=(1,)))
            self.assertAllEqual(fft_ops.ifftshift(shifted, axes=-1), fft_ops.ifftshift(shifted, axes=(1,)))
if __name__ == '__main__':
    test.main()