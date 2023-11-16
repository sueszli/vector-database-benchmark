import numpy
import pytest
import cupy
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy import testing

def _get_hermitian(xp, a, UPLO):
    if False:
        return 10
    if UPLO == 'U':
        return xp.triu(a) + xp.triu(a, 1).swapaxes(-2, -1).conj()
    else:
        return xp.tril(a) + xp.tril(a, -1).swapaxes(-2, -1).conj()

@testing.parameterize(*testing.product({'UPLO': ['U', 'L']}))
@pytest.mark.skipif(runtime.is_hip and driver.get_build_version() < 402, reason='eigensolver not added until ROCm 4.2.0')
class TestEigenvalue:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=0.001, atol=0.0001, contiguous_check=False)
    def test_eigh(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        if xp == numpy and dtype == numpy.float16:
            _dtype = 'f'
        else:
            _dtype = dtype
        if numpy.dtype(_dtype).kind == 'c':
            a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], _dtype)
        else:
            a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], _dtype)
        (w, v) = xp.linalg.eigh(a, UPLO=self.UPLO)
        A = _get_hermitian(xp, a, self.UPLO)
        if _dtype == numpy.float16:
            tol = 0.001
        else:
            tol = 1e-05
        testing.assert_allclose(A @ v, v @ xp.diag(w), atol=tol, rtol=tol)
        testing.assert_allclose(v @ v.swapaxes(-2, -1).conj(), xp.identity(A.shape[-1], _dtype), atol=tol, rtol=tol)
        if xp == numpy and dtype == numpy.float16:
            w = w.astype('e')
        return w

    @testing.for_all_dtypes(no_bool=True, no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=0.001, atol=0.0001, contiguous_check=False)
    def test_eigh_batched(self, xp, dtype):
        if False:
            print('Hello World!')
        a = xp.array([[[1, 0, 3], [0, 5, 0], [7, 0, 9]], [[3, 0, 3], [0, 7, 0], [7, 0, 11]]], dtype)
        (w, v) = xp.linalg.eigh(a, UPLO=self.UPLO)
        A = _get_hermitian(xp, a, self.UPLO)
        for i in range(a.shape[0]):
            testing.assert_allclose(A[i].dot(v[i]), w[i] * v[i], rtol=1e-05, atol=1e-05)
        return w

    @testing.for_dtypes('FD')
    @testing.numpy_cupy_allclose(rtol=0.001, atol=0.0001, contiguous_check=False)
    def test_eigh_complex_batched(self, xp, dtype):
        if False:
            return 10
        a = xp.array([[[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], [[0, 2j, 3], [4j, 4, 6j], [7, 8j, 8]]], dtype)
        (w, v) = xp.linalg.eigh(a, UPLO=self.UPLO)
        A = _get_hermitian(xp, a, self.UPLO)
        for i in range(a.shape[0]):
            testing.assert_allclose(A[i].dot(v[i]), w[i] * v[i], rtol=1e-05, atol=1e-05)
        return w

    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=0.001, atol=0.0001)
    def test_eigvalsh(self, xp, dtype):
        if False:
            while True:
                i = 10
        a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        return w

    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=0.001, atol=0.0001)
    def test_eigvalsh_batched(self, xp, dtype):
        if False:
            return 10
        a = xp.array([[[1, 0, 3], [0, 5, 0], [7, 0, 9]], [[3, 0, 3], [0, 7, 0], [7, 0, 11]]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        return w

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=0.001, atol=0.0001)
    def test_eigvalsh_complex(self, xp, dtype):
        if False:
            for i in range(10):
                print('nop')
        a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        return w

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=0.001, atol=0.0001)
    def test_eigvalsh_complex_batched(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        a = xp.array([[[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], [[0, 2j, 3], [4j, 4, 6j], [7, 8j, 8]]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        return w

@pytest.mark.parametrize('UPLO', ['U', 'L'])
@pytest.mark.parametrize('shape', [(0, 0), (2, 0, 0), (0, 3, 3)])
@pytest.mark.skipif(runtime.is_hip and driver.get_build_version() < 402, reason='eigensolver not added until ROCm 4.2.0')
class TestEigenvalueEmpty:

    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose()
    def test_eigh(self, xp, dtype, shape, UPLO):
        if False:
            for i in range(10):
                print('nop')
        a = xp.empty(shape, dtype)
        assert a.size == 0
        return xp.linalg.eigh(a, UPLO=UPLO)

    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose()
    def test_eigvalsh(self, xp, dtype, shape, UPLO):
        if False:
            i = 10
            return i + 15
        a = xp.empty(shape, dtype)
        assert a.size == 0
        return xp.linalg.eigvalsh(a, UPLO=UPLO)

@pytest.mark.parametrize('UPLO', ['U', 'L'])
@pytest.mark.parametrize('shape', [(), (3,), (2, 3), (4, 0), (2, 2, 3), (0, 2, 3)])
@pytest.mark.skipif(runtime.is_hip and driver.get_build_version() < 402, reason='eigensolver not added until ROCm 4.2.0')
class TestEigenvalueInvalid:

    def test_eigh_shape_error(self, UPLO, shape):
        if False:
            for i in range(10):
                print('nop')
        for xp in (numpy, cupy):
            a = xp.zeros(shape)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.eigh(a, UPLO)

    def test_eigvalsh_shape_error(self, UPLO, shape):
        if False:
            for i in range(10):
                print('nop')
        for xp in (numpy, cupy):
            a = xp.zeros(shape)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.eigvalsh(a, UPLO)