import numpy as np
from torch._numpy._ufuncs import *
from torch._numpy.testing import assert_allclose
from torch.testing._internal.common_utils import run_tests, TestCase

class TestUnaryUfuncs(TestCase):

    def test_absolute(self):
        if False:
            print('Hello World!')
        assert_allclose(np.absolute(0.5), absolute(0.5), atol=1e-14, check_dtype=False)

    def test_arccos(self):
        if False:
            return 10
        assert_allclose(np.arccos(0.5), arccos(0.5), atol=1e-14, check_dtype=False)

    def test_arccosh(self):
        if False:
            return 10
        assert_allclose(np.arccosh(1.5), arccosh(1.5), atol=1e-14, check_dtype=False)

    def test_arcsin(self):
        if False:
            print('Hello World!')
        assert_allclose(np.arcsin(0.5), arcsin(0.5), atol=1e-14, check_dtype=False)

    def test_arcsinh(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.arcsinh(0.5), arcsinh(0.5), atol=1e-14, check_dtype=False)

    def test_arctan(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.arctan(0.5), arctan(0.5), atol=1e-14, check_dtype=False)

    def test_arctanh(self):
        if False:
            return 10
        assert_allclose(np.arctanh(0.5), arctanh(0.5), atol=1e-14, check_dtype=False)

    def test_cbrt(self):
        if False:
            print('Hello World!')
        assert_allclose(np.cbrt(0.5), cbrt(0.5), atol=1e-14, check_dtype=False)

    def test_ceil(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.ceil(0.5), ceil(0.5), atol=1e-14, check_dtype=False)

    def test_conjugate(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.conjugate(0.5), conjugate(0.5), atol=1e-14, check_dtype=False)

    def test_cos(self):
        if False:
            while True:
                i = 10
        assert_allclose(np.cos(0.5), cos(0.5), atol=1e-14, check_dtype=False)

    def test_cosh(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.cosh(0.5), cosh(0.5), atol=1e-14, check_dtype=False)

    def test_deg2rad(self):
        if False:
            print('Hello World!')
        assert_allclose(np.deg2rad(0.5), deg2rad(0.5), atol=1e-14, check_dtype=False)

    def test_degrees(self):
        if False:
            print('Hello World!')
        assert_allclose(np.degrees(0.5), degrees(0.5), atol=1e-14, check_dtype=False)

    def test_exp(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.exp(0.5), exp(0.5), atol=1e-14, check_dtype=False)

    def test_exp2(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.exp2(0.5), exp2(0.5), atol=1e-14, check_dtype=False)

    def test_expm1(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.expm1(0.5), expm1(0.5), atol=1e-14, check_dtype=False)

    def test_fabs(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.fabs(0.5), fabs(0.5), atol=1e-14, check_dtype=False)

    def test_floor(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.floor(0.5), floor(0.5), atol=1e-14, check_dtype=False)

    def test_isfinite(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.isfinite(0.5), isfinite(0.5), atol=1e-14, check_dtype=False)

    def test_isinf(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.isinf(0.5), isinf(0.5), atol=1e-14, check_dtype=False)

    def test_isnan(self):
        if False:
            print('Hello World!')
        assert_allclose(np.isnan(0.5), isnan(0.5), atol=1e-14, check_dtype=False)

    def test_log(self):
        if False:
            return 10
        assert_allclose(np.log(0.5), log(0.5), atol=1e-14, check_dtype=False)

    def test_log10(self):
        if False:
            return 10
        assert_allclose(np.log10(0.5), log10(0.5), atol=1e-14, check_dtype=False)

    def test_log1p(self):
        if False:
            print('Hello World!')
        assert_allclose(np.log1p(0.5), log1p(0.5), atol=1e-14, check_dtype=False)

    def test_log2(self):
        if False:
            print('Hello World!')
        assert_allclose(np.log2(0.5), log2(0.5), atol=1e-14, check_dtype=False)

    def test_logical_not(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.logical_not(0.5), logical_not(0.5), atol=1e-14, check_dtype=False)

    def test_negative(self):
        if False:
            while True:
                i = 10
        assert_allclose(np.negative(0.5), negative(0.5), atol=1e-14, check_dtype=False)

    def test_positive(self):
        if False:
            print('Hello World!')
        assert_allclose(np.positive(0.5), positive(0.5), atol=1e-14, check_dtype=False)

    def test_rad2deg(self):
        if False:
            print('Hello World!')
        assert_allclose(np.rad2deg(0.5), rad2deg(0.5), atol=1e-14, check_dtype=False)

    def test_radians(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.radians(0.5), radians(0.5), atol=1e-14, check_dtype=False)

    def test_reciprocal(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.reciprocal(0.5), reciprocal(0.5), atol=1e-14, check_dtype=False)

    def test_rint(self):
        if False:
            print('Hello World!')
        assert_allclose(np.rint(0.5), rint(0.5), atol=1e-14, check_dtype=False)

    def test_sign(self):
        if False:
            print('Hello World!')
        assert_allclose(np.sign(0.5), sign(0.5), atol=1e-14, check_dtype=False)

    def test_signbit(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.signbit(0.5), signbit(0.5), atol=1e-14, check_dtype=False)

    def test_sin(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.sin(0.5), sin(0.5), atol=1e-14, check_dtype=False)

    def test_sinh(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.sinh(0.5), sinh(0.5), atol=1e-14, check_dtype=False)

    def test_sqrt(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.sqrt(0.5), sqrt(0.5), atol=1e-14, check_dtype=False)

    def test_square(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.square(0.5), square(0.5), atol=1e-14, check_dtype=False)

    def test_tan(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(np.tan(0.5), tan(0.5), atol=1e-14, check_dtype=False)

    def test_tanh(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.tanh(0.5), tanh(0.5), atol=1e-14, check_dtype=False)

    def test_trunc(self):
        if False:
            while True:
                i = 10
        assert_allclose(np.trunc(0.5), trunc(0.5), atol=1e-14, check_dtype=False)
if __name__ == '__main__':
    run_tests()