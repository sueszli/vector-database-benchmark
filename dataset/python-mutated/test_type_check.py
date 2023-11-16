import functools
from unittest import expectedFailure as xfail, skipIf as skipif
from pytest import raises as assert_raises
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO, TestCase, xpassIfTorchDynamo
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import common_type, iscomplex, iscomplexobj, isneginf, isposinf, isreal, isrealobj, nan_to_num, real_if_close
    from numpy.testing import assert_, assert_array_equal, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import common_type, iscomplex, iscomplexobj, isneginf, isposinf, isreal, isrealobj, nan_to_num, real_if_close
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal
skip = functools.partial(skipif, True)

def assert_all(x):
    if False:
        i = 10
        return i + 15
    assert_(np.all(x), x)

@xpassIfTorchDynamo
class TestCommonType(TestCase):

    def test_basic(self):
        if False:
            return 10
        ai32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        af16 = np.array([[1, 2], [3, 4]], dtype=np.float16)
        af32 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        af64 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        acs = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.csingle)
        acd = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.cdouble)
        assert_(common_type(ai32) == np.float64)
        assert_(common_type(af16) == np.float16)
        assert_(common_type(af32) == np.float32)
        assert_(common_type(af64) == np.float64)
        assert_(common_type(acs) == np.csingle)
        assert_(common_type(acd) == np.cdouble)

@xfail
class TestMintypecode(TestCase):

    def test_default_1(self):
        if False:
            i = 10
            return i + 15
        for itype in '1bcsuwil':
            assert_equal(mintypecode(itype), 'd')
        assert_equal(mintypecode('f'), 'f')
        assert_equal(mintypecode('d'), 'd')
        assert_equal(mintypecode('F'), 'F')
        assert_equal(mintypecode('D'), 'D')

    def test_default_2(self):
        if False:
            for i in range(10):
                print('nop')
        for itype in '1bcsuwil':
            assert_equal(mintypecode(itype + 'f'), 'f')
            assert_equal(mintypecode(itype + 'd'), 'd')
            assert_equal(mintypecode(itype + 'F'), 'F')
            assert_equal(mintypecode(itype + 'D'), 'D')
        assert_equal(mintypecode('ff'), 'f')
        assert_equal(mintypecode('fd'), 'd')
        assert_equal(mintypecode('fF'), 'F')
        assert_equal(mintypecode('fD'), 'D')
        assert_equal(mintypecode('df'), 'd')
        assert_equal(mintypecode('dd'), 'd')
        assert_equal(mintypecode('dF'), 'D')
        assert_equal(mintypecode('dD'), 'D')
        assert_equal(mintypecode('Ff'), 'F')
        assert_equal(mintypecode('Fd'), 'D')
        assert_equal(mintypecode('FF'), 'F')
        assert_equal(mintypecode('FD'), 'D')
        assert_equal(mintypecode('Df'), 'D')
        assert_equal(mintypecode('Dd'), 'D')
        assert_equal(mintypecode('DF'), 'D')
        assert_equal(mintypecode('DD'), 'D')

    def test_default_3(self):
        if False:
            return 10
        assert_equal(mintypecode('fdF'), 'D')
        assert_equal(mintypecode('fdD'), 'D')
        assert_equal(mintypecode('fFD'), 'D')
        assert_equal(mintypecode('dFD'), 'D')
        assert_equal(mintypecode('ifd'), 'd')
        assert_equal(mintypecode('ifF'), 'F')
        assert_equal(mintypecode('ifD'), 'D')
        assert_equal(mintypecode('idF'), 'D')
        assert_equal(mintypecode('idD'), 'D')

@xpassIfTorchDynamo
class TestIsscalar(TestCase):

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        assert_(np.isscalar(3))
        assert_(not np.isscalar([3]))
        assert_(not np.isscalar((3,)))
        assert_(np.isscalar(3j))
        assert_(np.isscalar(4.0))

class TestReal(TestCase):

    def test_real(self):
        if False:
            while True:
                i = 10
        y = np.random.rand(10)
        assert_array_equal(y, np.real(y))
        y = np.array(1)
        out = np.real(y)
        assert_array_equal(y, out)
        assert_(isinstance(out, np.ndarray))
        y = 1
        out = np.real(y)
        assert_equal(y, out)

    def test_cmplx(self):
        if False:
            return 10
        y = np.random.rand(10) + 1j * np.random.rand(10)
        assert_array_equal(y.real, np.real(y))
        y = np.array(1 + 1j)
        out = np.real(y)
        assert_array_equal(y.real, out)
        assert_(isinstance(out, np.ndarray))
        y = 1 + 1j
        out = np.real(y)
        assert_equal(1.0, out)

class TestImag(TestCase):

    def test_real(self):
        if False:
            return 10
        y = np.random.rand(10)
        assert_array_equal(0, np.imag(y))
        y = np.array(1)
        out = np.imag(y)
        assert_array_equal(0, out)
        assert_(isinstance(out, np.ndarray))
        y = 1
        out = np.imag(y)
        assert_equal(0, out)

    def test_cmplx(self):
        if False:
            while True:
                i = 10
        y = np.random.rand(10) + 1j * np.random.rand(10)
        assert_array_equal(y.imag, np.imag(y))
        y = np.array(1 + 1j)
        out = np.imag(y)
        assert_array_equal(y.imag, out)
        assert_(isinstance(out, np.ndarray))
        y = 1 + 1j
        out = np.imag(y)
        assert_equal(1.0, out)

class TestIscomplex(TestCase):

    def test_fail(self):
        if False:
            i = 10
            return i + 15
        z = np.array([-1, 0, 1])
        res = iscomplex(z)
        assert_(not np.sometrue(res, axis=0))

    def test_pass(self):
        if False:
            i = 10
            return i + 15
        z = np.array([-1j, 1, 0])
        res = iscomplex(z)
        assert_array_equal(res, [1, 0, 0])

class TestIsreal(TestCase):

    def test_pass(self):
        if False:
            i = 10
            return i + 15
        z = np.array([-1, 0, 1j])
        res = isreal(z)
        assert_array_equal(res, [1, 1, 0])

    def test_fail(self):
        if False:
            i = 10
            return i + 15
        z = np.array([-1j, 1, 0])
        res = isreal(z)
        assert_array_equal(res, [0, 1, 1])

    def test_isreal_real(self):
        if False:
            i = 10
            return i + 15
        z = np.array([-1, 0, 1])
        res = isreal(z)
        assert res.all()

class TestIscomplexobj(TestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        z = np.array([-1, 0, 1])
        assert_(not iscomplexobj(z))
        z = np.array([-1j, 0, -1])
        assert_(iscomplexobj(z))

    def test_scalar(self):
        if False:
            return 10
        assert_(not iscomplexobj(1.0))
        assert_(iscomplexobj(1 + 0j))

    def test_list(self):
        if False:
            return 10
        assert_(iscomplexobj([3, 1 + 0j, True]))
        assert_(not iscomplexobj([3, 1, True]))

class TestIsrealobj(TestCase):

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        z = np.array([-1, 0, 1])
        assert_(isrealobj(z))
        z = np.array([-1j, 0, -1])
        assert_(not isrealobj(z))

class TestIsnan(TestCase):

    def test_goodvalues(self):
        if False:
            i = 10
            return i + 15
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isnan(z) == 0
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        if False:
            return 10
        assert_all(np.isnan(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        if False:
            print('Hello World!')
        assert_all(np.isnan(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        if False:
            return 10
        assert_all(np.isnan(np.array((0.0,)) / 0.0) == 1)

    def test_integer(self):
        if False:
            print('Hello World!')
        assert_all(np.isnan(1) == 0)

    def test_complex(self):
        if False:
            while True:
                i = 10
        assert_all(np.isnan(1 + 1j) == 0)

    def test_complex1(self):
        if False:
            while True:
                i = 10
        assert_all(np.isnan(np.array(0 + 0j) / 0.0) == 1)

class TestIsfinite(TestCase):

    def test_goodvalues(self):
        if False:
            print('Hello World!')
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isfinite(z) == 1
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        if False:
            for i in range(10):
                print('nop')
        assert_all(np.isfinite(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        if False:
            for i in range(10):
                print('nop')
        assert_all(np.isfinite(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        if False:
            for i in range(10):
                print('nop')
        assert_all(np.isfinite(np.array((0.0,)) / 0.0) == 0)

    def test_integer(self):
        if False:
            print('Hello World!')
        assert_all(np.isfinite(1) == 1)

    def test_complex(self):
        if False:
            return 10
        assert_all(np.isfinite(1 + 1j) == 1)

    def test_complex1(self):
        if False:
            return 10
        assert_all(np.isfinite(np.array(1 + 1j) / 0.0) == 0)

class TestIsinf(TestCase):

    def test_goodvalues(self):
        if False:
            while True:
                i = 10
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isinf(z) == 0
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        if False:
            print('Hello World!')
        assert_all(np.isinf(np.array((1.0,)) / 0.0) == 1)

    def test_posinf_scalar(self):
        if False:
            print('Hello World!')
        assert_all(np.isinf(np.array(1.0) / 0.0) == 1)

    def test_neginf(self):
        if False:
            print('Hello World!')
        assert_all(np.isinf(np.array((-1.0,)) / 0.0) == 1)

    def test_neginf_scalar(self):
        if False:
            print('Hello World!')
        assert_all(np.isinf(np.array(-1.0) / 0.0) == 1)

    def test_ind(self):
        if False:
            for i in range(10):
                print('nop')
        assert_all(np.isinf(np.array((0.0,)) / 0.0) == 0)

class TestIsposinf(TestCase):

    def test_generic(self):
        if False:
            while True:
                i = 10
        vals = isposinf(np.array((-1.0, 0, 1)) / 0.0)
        assert_(vals[0] == 0)
        assert_(vals[1] == 0)
        assert_(vals[2] == 1)

class TestIsneginf(TestCase):

    def test_generic(self):
        if False:
            while True:
                i = 10
        vals = isneginf(np.array((-1.0, 0, 1)) / 0.0)
        assert_(vals[0] == 1)
        assert_(vals[1] == 0)
        assert_(vals[2] == 0)

class TestNanToNum(TestCase):

    def test_generic(self):
        if False:
            print('Hello World!')
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0)
        assert_all(vals[0] < -10000000000.0) and assert_all(np.isfinite(vals[0]))
        assert_(vals[1] == 0)
        assert_all(vals[2] > 10000000000.0) and assert_all(np.isfinite(vals[2]))
        assert isinstance(vals, np.ndarray)
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=10, posinf=20, neginf=30)
        assert_equal(vals, [30, 10, 20])
        assert_all(np.isfinite(vals[[0, 2]]))
        assert isinstance(vals, np.ndarray)

    def test_array(self):
        if False:
            return 10
        vals = nan_to_num([1])
        assert_array_equal(vals, np.array([1], int))
        assert isinstance(vals, np.ndarray)
        vals = nan_to_num([1], nan=10, posinf=20, neginf=30)
        assert_array_equal(vals, np.array([1], int))
        assert isinstance(vals, np.ndarray)

    @skip(reason='we return OD arrays not scalars')
    def test_integer(self):
        if False:
            for i in range(10):
                print('nop')
        vals = nan_to_num(1)
        assert_all(vals == 1)
        assert isinstance(vals, np.int_)
        vals = nan_to_num(1, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1)
        assert isinstance(vals, np.int_)

    @skip(reason='we return OD arrays not scalars')
    def test_float(self):
        if False:
            i = 10
            return i + 15
        vals = nan_to_num(1.0)
        assert_all(vals == 1.0)
        assert_equal(type(vals), np.float_)
        vals = nan_to_num(1.1, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1.1)
        assert_equal(type(vals), np.float_)

    @skip(reason='we return OD arrays not scalars')
    def test_complex_good(self):
        if False:
            for i in range(10):
                print('nop')
        vals = nan_to_num(1 + 1j)
        assert_all(vals == 1 + 1j)
        assert isinstance(vals, np.complex_)
        vals = nan_to_num(1 + 1j, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1 + 1j)
        assert_equal(type(vals), np.complex_)

    @skip(reason='we return OD arrays not scalars')
    def test_complex_bad(self):
        if False:
            return 10
        v = 1 + 1j
        v += np.array(0 + 1j) / 0.0
        vals = nan_to_num(v)
        assert_all(np.isfinite(vals))
        assert_equal(type(vals), np.complex_)

    @skip(reason='we return OD arrays not scalars')
    def test_complex_bad2(self):
        if False:
            for i in range(10):
                print('nop')
        v = 1 + 1j
        v += np.array(-1 + 1j) / 0.0
        vals = nan_to_num(v)
        assert_all(np.isfinite(vals))
        assert_equal(type(vals), np.complex_)

    def test_do_not_rewrite_previous_keyword(self):
        if False:
            while True:
                i = 10
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=np.inf, posinf=999)
        assert_all(np.isfinite(vals[[0, 2]]))
        assert_all(vals[0] < -10000000000.0)
        assert_equal(vals[[1, 2]], [np.inf, 999])
        assert isinstance(vals, np.ndarray)

class TestRealIfClose(TestCase):

    def test_basic(self):
        if False:
            return 10
        a = np.random.rand(10)
        b = real_if_close(a + 1e-15j)
        assert_all(isrealobj(b))
        assert_array_equal(a, b)
        b = real_if_close(a + 1e-07j)
        assert_all(iscomplexobj(b))
        b = real_if_close(a + 1e-07j, tol=1e-06)
        assert_all(isrealobj(b))

@xfail
class TestArrayConversion(TestCase):

    def test_asfarray(self):
        if False:
            i = 10
            return i + 15
        a = asfarray(np.array([1, 2, 3]))
        assert_equal(a.__class__, np.ndarray)
        assert_(np.issubdtype(a.dtype, np.floating))
        assert_raises(TypeError, asfarray, np.array([1, 2, 3]), dtype=np.array(1.0))
if __name__ == '__main__':
    run_tests()