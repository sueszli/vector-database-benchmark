import numpy as np
import pytest
from cupyx.scipy import fft as cp_fft
from cupy import testing
try:
    import scipy.fft as scipy_fft
except ImportError:
    scipy_fft = None
scipy_cplx_bug = not cp_fft._scipy_150
if cp_fft._scipy_160:
    all_dct_norms = [None, 'ortho', 'forward', 'backward']
else:
    all_dct_norms = [None, 'ortho']

@testing.parameterize(*testing.product({'n': [None, 0, 5, 15], 'type': [1, 2, 3, 4], 'shape': [(9,), (10,), (10, 9)], 'axis': [-1, 0], 'norm': ['ortho'], 'overwrite_x': [False], 'function': ['dct', 'dst', 'idct', 'idst']}) + testing.product({'n': [None, 15], 'type': [2, 3], 'shape': [(10, 9)], 'axis': [-1, 0], 'norm': all_dct_norms, 'overwrite_x': [False, True], 'function': ['dct', 'dst', 'idct', 'idst']}))
@testing.with_requires('scipy>=1.4')
class TestDctDst:

    def _run_transform(self, dct_func, xp, dtype):
        if False:
            while True:
                i = 10
        x = testing.shaped_random(self.shape, xp, dtype)
        if scipy_cplx_bug and x.dtype.kind == 'c':
            return x
        x_orig = x.copy()
        kwargs = dict(type=self.type, n=self.n, axis=self.axis, norm=self.norm, overwrite_x=self.overwrite_x)
        if self.type in [1, 4]:
            if xp != np:
                with pytest.raises(NotImplementedError):
                    dct_func(x, **kwargs)
            return xp.zeros([])
        out = dct_func(x, **kwargs)
        if not self.overwrite_x:
            testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=0.0001, atol=1e-05, accept_error=ValueError, contiguous_check=False)
    def test_dct(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        fft_func = getattr(scp.fft, self.function)
        return self._run_transform(fft_func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=0.0001, atol=1e-05, accept_error=ValueError, contiguous_check=False)
    def test_dct_backend(self, xp, dtype):
        if False:
            return 10
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            fft_func = getattr(scipy_fft, self.function)
            return self._run_transform(fft_func, xp, dtype)

@testing.parameterize(*testing.product({'shape': [(3, 4)], 'type': [2, 3], 's': [None, (1, 5), (-1, -1), (0, 5), (1.5, 2.5)], 'axes': [None, (-2, -1), (-1, -2), (0,)], 'norm': ['ortho'], 'overwrite_x': [False], 'function': ['dctn', 'dstn', 'idctn', 'idstn']}) + testing.product({'shape': [(2, 3, 4)], 'type': [2, 3], 's': [None, (1, 5), (1, 4, 10), (2, 2, 2, 2)], 'axes': [None, (-2, -1), (-1, -2, -3)], 'norm': ['ortho'], 'overwrite_x': [False], 'function': ['dctn', 'dstn', 'idctn', 'idstn']}) + testing.product({'shape': [(2, 3, 4, 5)], 'type': [2, 3], 's': [None], 'axes': [None], 'norm': all_dct_norms, 'overwrite_x': [True, False], 'function': ['dctn', 'dstn', 'idctn', 'idstn']}))
@testing.with_requires('scipy>=1.4')
class TestDctnDstn:

    def _run_transform(self, dct_func, xp, dtype):
        if False:
            return 10
        x = testing.shaped_random(self.shape, xp, dtype)
        if scipy_cplx_bug and x.dtype.kind == 'c':
            return x
        x_orig = x.copy()
        out = dct_func(x, type=self.type, s=self.s, axes=self.axes, norm=self.norm, overwrite_x=self.overwrite_x)
        if not self.overwrite_x:
            testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=0.0001, atol=1e-05, accept_error=ValueError, contiguous_check=False)
    def test_dctn(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        fft_func = getattr(scp.fft, self.function)
        return self._run_transform(fft_func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=0.0001, atol=1e-05, accept_error=ValueError, contiguous_check=False)
    def test_dctn_backend(self, xp, dtype):
        if False:
            for i in range(10):
                print('nop')
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            fft_func = getattr(scipy_fft, self.function)
            return self._run_transform(fft_func, xp, dtype)