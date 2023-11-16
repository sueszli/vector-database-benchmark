import warnings
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, assert_warns, assert_array_equal, temppath, IS_MUSL
from numpy._core.tests._locales import CommaDecimalPointLocale
LD_INFO = np.finfo(np.longdouble)
longdouble_longer_than_double = LD_INFO.eps < np.finfo(np.double).eps
_o = 1 + LD_INFO.eps
string_to_longdouble_inaccurate = _o != np.longdouble(str(_o))
del _o

def test_scalar_extraction():
    if False:
        i = 10
        return i + 15
    "Confirm that extracting a value doesn't convert to python float"
    o = 1 + LD_INFO.eps
    a = np.array([o, o, o])
    assert_equal(a[1], o)
repr_precision = len(repr(np.longdouble(0.1)))

@pytest.mark.skipif(IS_MUSL, reason='test flaky on musllinux')
@pytest.mark.skipif(LD_INFO.precision + 2 >= repr_precision, reason='repr precision not enough to show eps')
def test_str_roundtrip():
    if False:
        i = 10
        return i + 15
    o = 1 + LD_INFO.eps
    assert_equal(np.longdouble(str(o)), o, 'str was %s' % str(o))

@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
def test_str_roundtrip_bytes():
    if False:
        i = 10
        return i + 15
    o = 1 + LD_INFO.eps
    assert_equal(np.longdouble(str(o).encode('ascii')), o)

@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
@pytest.mark.parametrize('strtype', (np.str_, np.bytes_, str, bytes))
def test_array_and_stringlike_roundtrip(strtype):
    if False:
        return 10
    '\n    Test that string representations of long-double roundtrip both\n    for array casting and scalar coercion, see also gh-15608.\n    '
    o = 1 + LD_INFO.eps
    if strtype in (np.bytes_, bytes):
        o_str = strtype(str(o).encode('ascii'))
    else:
        o_str = strtype(str(o))
    assert o == np.longdouble(o_str)
    o_strarr = np.asarray([o] * 3, dtype=strtype)
    assert (o == o_strarr.astype(np.longdouble)).all()
    assert (o_strarr == o_str).all()
    assert (np.asarray([o] * 3).astype(strtype) == o_str).all()

def test_bogus_string():
    if False:
        print('Hello World!')
    assert_raises(ValueError, np.longdouble, 'spam')
    assert_raises(ValueError, np.longdouble, '1.0 flub')

@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
def test_fromstring():
    if False:
        print('Hello World!')
    o = 1 + LD_INFO.eps
    s = (' ' + str(o)) * 5
    a = np.array([o] * 5)
    assert_equal(np.fromstring(s, sep=' ', dtype=np.longdouble), a, err_msg="reading '%s'" % s)

def test_fromstring_complex():
    if False:
        return 10
    for ctype in ['complex', 'cdouble']:
        assert_equal(np.fromstring('1, 2 ,  3  ,4', sep=',', dtype=ctype), np.array([1.0, 2.0, 3.0, 4.0]))
        assert_equal(np.fromstring('1j, -2j,  3j, 4e1j', sep=',', dtype=ctype), np.array([1j, -2j, 3j, 40j]))
        assert_equal(np.fromstring('1+1j,2-2j, -3+3j,  -4e1+4j', sep=',', dtype=ctype), np.array([1.0 + 1j, 2.0 - 2j, -3.0 + 3j, -40.0 + 4j]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1+2 j,3', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1+ 2j,3', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1 +2j,3', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1+j', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1+', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1j+1', dtype=ctype, sep=','), np.array([1j]))

def test_fromstring_bogus():
    if False:
        print('Hello World!')
    with assert_warns(DeprecationWarning):
        assert_equal(np.fromstring('1. 2. 3. flop 4.', dtype=float, sep=' '), np.array([1.0, 2.0, 3.0]))

def test_fromstring_empty():
    if False:
        i = 10
        return i + 15
    with assert_warns(DeprecationWarning):
        assert_equal(np.fromstring('xxxxx', sep='x'), np.array([]))

def test_fromstring_missing():
    if False:
        return 10
    with assert_warns(DeprecationWarning):
        assert_equal(np.fromstring('1xx3x4x5x6', sep='x'), np.array([1]))

class TestFileBased:
    ldbl = 1 + LD_INFO.eps
    tgt = np.array([ldbl] * 5)
    out = ''.join([str(t) + '\n' for t in tgt])

    def test_fromfile_bogus(self):
        if False:
            return 10
        with temppath() as path:
            with open(path, 'w') as f:
                f.write('1. 2. 3. flop 4.\n')
            with assert_warns(DeprecationWarning):
                res = np.fromfile(path, dtype=float, sep=' ')
        assert_equal(res, np.array([1.0, 2.0, 3.0]))

    def test_fromfile_complex(self):
        if False:
            print('Hello World!')
        for ctype in ['complex', 'cdouble']:
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1, 2 ,  3  ,4\n')
                res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0, 2.0, 3.0, 4.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1j, -2j,  3j, 4e1j\n')
                res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1j, -2j, 3j, 40j]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+1j,2-2j, -3+3j,  -4e1+4j\n')
                res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0 + 1j, 2.0 - 2j, -3.0 + 3j, -40.0 + 4j]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+2 j,3\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+ 2j,3\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1 +2j,3\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+j\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1j+1\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1j]))

    @pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
    def test_fromfile(self):
        if False:
            for i in range(10):
                print('nop')
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            res = np.fromfile(path, dtype=np.longdouble, sep='\n')
        assert_equal(res, self.tgt)

    @pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
    def test_genfromtxt(self):
        if False:
            while True:
                i = 10
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            res = np.genfromtxt(path, dtype=np.longdouble)
        assert_equal(res, self.tgt)

    @pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
    def test_loadtxt(self):
        if False:
            print('Hello World!')
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            res = np.loadtxt(path, dtype=np.longdouble)
        assert_equal(res, self.tgt)

    @pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
    def test_tofile_roundtrip(self):
        if False:
            print('Hello World!')
        with temppath() as path:
            self.tgt.tofile(path, sep=' ')
            res = np.fromfile(path, dtype=np.longdouble, sep=' ')
        assert_equal(res, self.tgt)

def test_str_exact():
    if False:
        for i in range(10):
            print('nop')
    o = 1 + LD_INFO.eps
    assert_(str(o) != '1')

@pytest.mark.skipif(longdouble_longer_than_double, reason='BUG #2376')
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
def test_format():
    if False:
        return 10
    o = 1 + LD_INFO.eps
    assert_('{0:.40g}'.format(o) != '1')

@pytest.mark.skipif(longdouble_longer_than_double, reason='BUG #2376')
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
def test_percent():
    if False:
        for i in range(10):
            print('nop')
    o = 1 + LD_INFO.eps
    assert_('%.40g' % o != '1')

@pytest.mark.skipif(longdouble_longer_than_double, reason='array repr problem')
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
def test_array_repr():
    if False:
        return 10
    o = 1 + LD_INFO.eps
    a = np.array([o])
    b = np.array([1], dtype=np.longdouble)
    if not np.all(a != b):
        raise ValueError('precision loss creating arrays')
    assert_(repr(a) != repr(b))

class TestCommaDecimalPointLocale(CommaDecimalPointLocale):

    def test_str_roundtrip_foreign(self):
        if False:
            i = 10
            return i + 15
        o = 1.5
        assert_equal(o, np.longdouble(str(o)))

    def test_fromstring_foreign_repr(self):
        if False:
            while True:
                i = 10
        f = 1.234
        a = np.fromstring(repr(f), dtype=float, sep=' ')
        assert_equal(a[0], f)

    def test_fromstring_best_effort_float(self):
        if False:
            i = 10
            return i + 15
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1,234', dtype=float, sep=' '), np.array([1.0]))

    def test_fromstring_best_effort(self):
        if False:
            return 10
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1,234', dtype=np.longdouble, sep=' '), np.array([1.0]))

    def test_fromstring_foreign(self):
        if False:
            i = 10
            return i + 15
        s = '1.234'
        a = np.fromstring(s, dtype=np.longdouble, sep=' ')
        assert_equal(a[0], np.longdouble(s))

    def test_fromstring_foreign_sep(self):
        if False:
            print('Hello World!')
        a = np.array([1, 2, 3, 4])
        b = np.fromstring('1,2,3,4,', dtype=np.longdouble, sep=',')
        assert_array_equal(a, b)

    def test_fromstring_foreign_value(self):
        if False:
            for i in range(10):
                print('nop')
        with assert_warns(DeprecationWarning):
            b = np.fromstring('1,234', dtype=np.longdouble, sep=' ')
            assert_array_equal(b[0], 1)

@pytest.mark.parametrize('int_val', [2 ** 1024, 0])
def test_longdouble_from_int(int_val):
    if False:
        while True:
            i = 10
    str_val = str(int_val)
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', RuntimeWarning)
        assert np.longdouble(int_val) == np.longdouble(str_val)
        if np.allclose(np.finfo(np.longdouble).max, np.finfo(np.double).max) and w:
            assert w[0].category is RuntimeWarning

@pytest.mark.parametrize('bool_val', [True, False])
def test_longdouble_from_bool(bool_val):
    if False:
        for i in range(10):
            print('nop')
    assert np.longdouble(bool_val) == np.longdouble(int(bool_val))

@pytest.mark.skipif(not (IS_MUSL and platform.machine() == 'x86_64'), reason='only need to run on musllinux_x86_64')
def test_musllinux_x86_64_signature():
    if False:
        return 10
    known_sigs = [b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf']
    sig = np.longdouble(-1.0) / np.longdouble(10.0)
    sig = sig.view(sig.dtype.newbyteorder('<')).tobytes()[:10]
    assert sig in known_sigs

def test_eps_positive():
    if False:
        while True:
            i = 10
    assert np.finfo(np.longdouble).eps > 0.0