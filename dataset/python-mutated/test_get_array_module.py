import unittest
from cupy import testing
import cupyx.scipy.special

@testing.with_requires('scipy')
class TestSpecial(unittest.TestCase):

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def test_get_array_module(self, xp, scp, dtype):
        if False:
            return 10
        import scipy.special
        a = testing.shaped_arange((2, 3), xp, dtype)
        module = cupyx.scipy.get_array_module(a)
        assert module is scp
        return module.special.j0(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def test_get_array_module_multiple_parameters(self, xp, scp, dtype):
        if False:
            for i in range(10):
                print('nop')
        import scipy.special
        a = testing.shaped_arange((2, 3), xp, dtype)
        module = cupyx.scipy.get_array_module(a, a)
        assert module is scp
        return module.special.j1(a)