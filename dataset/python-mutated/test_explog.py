import numpy
import pytest
from cupy import testing

class TestExplog:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-05)
    def check_unary(self, name, xp, dtype, no_complex=False):
        if False:
            return 10
        if no_complex:
            if numpy.dtype(dtype).kind == 'c':
                return xp.array(True)
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-05)
    def check_binary(self, name, xp, dtype, no_complex=False):
        if False:
            print('Hello World!')
        if no_complex:
            if numpy.dtype(dtype).kind == 'c':
                return xp.array(True)
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    def test_exp(self):
        if False:
            print('Hello World!')
        self.check_unary('exp')

    def test_expm1(self):
        if False:
            print('Hello World!')
        self.check_unary('expm1')

    def test_exp2(self):
        if False:
            while True:
                i = 10
        self.check_unary('exp2')

    def test_log(self):
        if False:
            print('Hello World!')
        with numpy.errstate(divide='ignore'):
            self.check_unary('log')

    def test_log10(self):
        if False:
            print('Hello World!')
        with numpy.errstate(divide='ignore'):
            self.check_unary('log10')

    def test_log2(self):
        if False:
            print('Hello World!')
        with numpy.errstate(divide='ignore'):
            self.check_unary('log2')

    def test_log1p(self):
        if False:
            return 10
        self.check_unary('log1p')

    def test_logaddexp(self):
        if False:
            return 10
        self.check_binary('logaddexp', no_complex=True)

    @pytest.mark.parametrize('val', [numpy.inf, -numpy.inf])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_logaddexp_infinities(self, xp, dtype, val):
        if False:
            return 10
        a = xp.full((2, 3), val, dtype=dtype)
        return xp.logaddexp(a, a)

    def test_logaddexp2(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_binary('logaddexp2', no_complex=True)

    @pytest.mark.parametrize('val', [numpy.inf, -numpy.inf])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_logaddexp2_infinities(self, xp, dtype, val):
        if False:
            i = 10
            return i + 15
        a = xp.full((2, 3), val, dtype=dtype)
        return xp.logaddexp2(a, a)