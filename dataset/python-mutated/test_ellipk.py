from cupy import testing
try:
    import cupyx.scipy.special
except ImportError:
    pass

@testing.with_requires('scipy')
class TestEllipk:

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15)
    def test_basic_ellipk(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.linspace(1e-14, 1, 101)
        return scp.special.ellipk(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15)
    def test_basic_ellipkm1(self, xp, scp):
        if False:
            while True:
                i = 10
        x = xp.linspace(1e-14, 1, 101)
        return scp.special.ellipkm1(1.0 / x)

@testing.with_requires('scipy')
class TestEllipj:

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13)
    def test_basic(self, xp, scp):
        if False:
            print('Hello World!')
        el = scp.special.ellipj(0.2, 0)
        return el