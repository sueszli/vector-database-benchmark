import cupyx.scipy.special
from cupy import testing
from cupy.testing import numpy_cupy_allclose

@testing.with_requires('scipy')
class TestExp1:

    @numpy_cupy_allclose(scipy_name='scp')
    def test_exp1_zero(self, xp, scp):
        if False:
            i = 10
            return i + 15
        return scp.special.exp1(0.0)

    @numpy_cupy_allclose(scipy_name='scp')
    def test_exp1_negative(self, xp, scp):
        if False:
            while True:
                i = 10
        return scp.special.exp1(-4.2)

    @numpy_cupy_allclose(scipy_name='scp')
    def test_exp1_large_negative(self, xp, scp):
        if False:
            return 10
        return scp.special.exp1(-1000)

    @numpy_cupy_allclose(scipy_name='scp')
    def test_exp1_positive(self, xp, scp):
        if False:
            print('Hello World!')
        return scp.special.exp1(4.2)

    @numpy_cupy_allclose(scipy_name='scp')
    def test_exp1_large_positive(self, xp, scp):
        if False:
            print('Hello World!')
        return scp.special.exp1(1000)

    @testing.for_dtypes(['f'])
    @numpy_cupy_allclose(scipy_name='scp', rtol=1e-05)
    def test_exp1_linspace_float32(self, xp, scp, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = xp.linspace(-10, 60, 1000, dtype=dtype)
        return scp.special.exp1(x)

    @testing.for_dtypes(['d'])
    @numpy_cupy_allclose(scipy_name='scp', rtol=1e-12)
    def test_exp1_linspace_float64(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        x = xp.linspace(-10, 100, 1000, dtype=dtype)
        return scp.special.exp1(x)