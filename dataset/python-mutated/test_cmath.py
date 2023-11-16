from test.support import requires_IEEE_754, cpython_only
from test.test_math import parse_testfile, test_file
import test.test_math as test_math
import unittest
import cmath, math
from cmath import phase, polar, rect, pi
import platform
import sys
INF = float('inf')
NAN = float('nan')
complex_zeros = [complex(x, y) for x in [0.0, -0.0] for y in [0.0, -0.0]]
complex_infinities = [complex(x, y) for (x, y) in [(INF, 0.0), (INF, 2.3), (INF, INF), (2.3, INF), (0.0, INF), (-0.0, INF), (-2.3, INF), (-INF, INF), (-INF, 2.3), (-INF, 0.0), (-INF, -0.0), (-INF, -2.3), (-INF, -INF), (-2.3, -INF), (-0.0, -INF), (0.0, -INF), (2.3, -INF), (INF, -INF), (INF, -2.3), (INF, -0.0)]]
complex_nans = [complex(x, y) for (x, y) in [(NAN, -INF), (NAN, -2.3), (NAN, -0.0), (NAN, 0.0), (NAN, 2.3), (NAN, INF), (-INF, NAN), (-2.3, NAN), (-0.0, NAN), (0.0, NAN), (2.3, NAN), (INF, NAN)]]

class CMathTests(unittest.TestCase):
    test_functions = [getattr(cmath, fname) for fname in ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh', 'exp', 'log', 'log10', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']]
    test_functions.append(lambda x: cmath.log(x, 1729.0 + 0j))
    test_functions.append(lambda x: cmath.log(14.0 - 27j, x))

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_values = open(test_file, encoding='utf-8')

    def tearDown(self):
        if False:
            return 10
        self.test_values.close()

    def assertFloatIdentical(self, x, y):
        if False:
            return 10
        'Fail unless floats x and y are identical, in the sense that:\n        (1) both x and y are nans, or\n        (2) both x and y are infinities, with the same sign, or\n        (3) both x and y are zeros, with the same sign, or\n        (4) x and y are both finite and nonzero, and x == y\n\n        '
        msg = 'floats {!r} and {!r} are not identical'
        if math.isnan(x) or math.isnan(y):
            if math.isnan(x) and math.isnan(y):
                return
        elif x == y:
            if x != 0.0:
                return
            elif math.copysign(1.0, x) == math.copysign(1.0, y):
                return
            else:
                msg += ': zeros have different signs'
        self.fail(msg.format(x, y))

    def assertComplexIdentical(self, x, y):
        if False:
            while True:
                i = 10
        'Fail unless complex numbers x and y have equal values and signs.\n\n        In particular, if x and y both have real (or imaginary) part\n        zero, but the zeros have different signs, this test will fail.\n\n        '
        self.assertFloatIdentical(x.real, y.real)
        self.assertFloatIdentical(x.imag, y.imag)

    def rAssertAlmostEqual(self, a, b, rel_err=2e-15, abs_err=5e-323, msg=None):
        if False:
            i = 10
            return i + 15
        'Fail if the two floating-point numbers are not almost equal.\n\n        Determine whether floating-point values a and b are equal to within\n        a (small) rounding error.  The default values for rel_err and\n        abs_err are chosen to be suitable for platforms where a float is\n        represented by an IEEE 754 double.  They allow an error of between\n        9 and 19 ulps.\n        '
        if math.isnan(a):
            if math.isnan(b):
                return
            self.fail(msg or '{!r} should be nan'.format(b))
        if math.isinf(a):
            if a == b:
                return
            self.fail(msg or 'finite result where infinity expected: expected {!r}, got {!r}'.format(a, b))
        if not a and (not b):
            if math.copysign(1.0, a) != math.copysign(1.0, b):
                self.fail(msg or 'zero has wrong sign: expected {!r}, got {!r}'.format(a, b))
        try:
            absolute_error = abs(b - a)
        except OverflowError:
            pass
        else:
            if absolute_error <= max(abs_err, rel_err * abs(a)):
                return
        self.fail(msg or '{!r} and {!r} are not sufficiently close'.format(a, b))

    def test_constants(self):
        if False:
            while True:
                i = 10
        e_expected = 2.718281828459045
        pi_expected = 3.141592653589793
        self.assertAlmostEqual(cmath.pi, pi_expected, places=9, msg='cmath.pi is {}; should be {}'.format(cmath.pi, pi_expected))
        self.assertAlmostEqual(cmath.e, e_expected, places=9, msg='cmath.e is {}; should be {}'.format(cmath.e, e_expected))

    def test_infinity_and_nan_constants(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(cmath.inf.real, math.inf)
        self.assertEqual(cmath.inf.imag, 0.0)
        self.assertEqual(cmath.infj.real, 0.0)
        self.assertEqual(cmath.infj.imag, math.inf)
        self.assertTrue(math.isnan(cmath.nan.real))
        self.assertEqual(cmath.nan.imag, 0.0)
        self.assertEqual(cmath.nanj.real, 0.0)
        self.assertTrue(math.isnan(cmath.nanj.imag))
        self.assertEqual(repr(cmath.inf), 'inf')
        self.assertEqual(repr(cmath.infj), 'infj')
        self.assertEqual(repr(cmath.nan), 'nan')
        self.assertEqual(repr(cmath.nanj), 'nanj')

    def test_user_object(self):
        if False:
            while True:
                i = 10
        cx_arg = 4.419414439 + 1.497100113j
        flt_arg = -6.131677725
        non_complexes = ['not complex', 1, 5, 2.0, None, object(), NotImplemented]

        class MyComplex(object):

            def __init__(self, value):
                if False:
                    i = 10
                    return i + 15
                self.value = value

            def __complex__(self):
                if False:
                    return 10
                return self.value

        class MyComplexOS:

            def __init__(self, value):
                if False:
                    return 10
                self.value = value

            def __complex__(self):
                if False:
                    print('Hello World!')
                return self.value

        class SomeException(Exception):
            pass

        class MyComplexException(object):

            def __complex__(self):
                if False:
                    while True:
                        i = 10
                raise SomeException

        class MyComplexExceptionOS:

            def __complex__(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise SomeException

        class NeitherComplexNorFloat(object):
            pass

        class NeitherComplexNorFloatOS:
            pass

        class Index:

            def __int__(self):
                if False:
                    print('Hello World!')
                return 2

            def __index__(self):
                if False:
                    return 10
                return 2

        class MyInt:

            def __int__(self):
                if False:
                    i = 10
                    return i + 15
                return 2

        class FloatAndComplex(object):

            def __float__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return flt_arg

            def __complex__(self):
                if False:
                    print('Hello World!')
                return cx_arg

        class FloatAndComplexOS:

            def __float__(self):
                if False:
                    return 10
                return flt_arg

            def __complex__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return cx_arg

        class JustFloat(object):

            def __float__(self):
                if False:
                    return 10
                return flt_arg

        class JustFloatOS:

            def __float__(self):
                if False:
                    i = 10
                    return i + 15
                return flt_arg
        for f in self.test_functions:
            self.assertEqual(f(MyComplex(cx_arg)), f(cx_arg))
            self.assertEqual(f(MyComplexOS(cx_arg)), f(cx_arg))
            self.assertEqual(f(FloatAndComplex()), f(cx_arg))
            self.assertEqual(f(FloatAndComplexOS()), f(cx_arg))
            self.assertEqual(f(JustFloat()), f(flt_arg))
            self.assertEqual(f(JustFloatOS()), f(flt_arg))
            self.assertEqual(f(Index()), f(int(Index())))
            self.assertRaises(TypeError, f, NeitherComplexNorFloat())
            self.assertRaises(TypeError, f, MyInt())
            self.assertRaises(Exception, f, NeitherComplexNorFloatOS())
            for bad_complex in non_complexes:
                self.assertRaises(TypeError, f, MyComplex(bad_complex))
                self.assertRaises(TypeError, f, MyComplexOS(bad_complex))
            self.assertRaises(SomeException, f, MyComplexException())
            self.assertRaises(SomeException, f, MyComplexExceptionOS())

    def test_input_type(self):
        if False:
            for i in range(10):
                print('nop')
        for f in self.test_functions:
            for arg in [2, 2.0]:
                self.assertEqual(f(arg), f(arg.__float__()))
        for f in self.test_functions:
            for arg in ['a', 'long_string', '0', '1j', '']:
                self.assertRaises(TypeError, f, arg)

    def test_cmath_matches_math(self):
        if False:
            i = 10
            return i + 15
        test_values = [0.01, 0.1, 0.2, 0.5, 0.9, 0.99]
        unit_interval = test_values + [-x for x in test_values] + [0.0, 1.0, -1.0]
        positive = test_values + [1.0] + [1.0 / x for x in test_values]
        nonnegative = [0.0] + positive
        real_line = [0.0] + positive + [-x for x in positive]
        test_functions = {'acos': unit_interval, 'asin': unit_interval, 'atan': real_line, 'cos': real_line, 'cosh': real_line, 'exp': real_line, 'log': positive, 'log10': positive, 'sin': real_line, 'sinh': real_line, 'sqrt': nonnegative, 'tan': real_line, 'tanh': real_line}
        for (fn, values) in test_functions.items():
            float_fn = getattr(math, fn)
            complex_fn = getattr(cmath, fn)
            for v in values:
                z = complex_fn(v)
                self.rAssertAlmostEqual(float_fn(v), z.real)
                self.assertEqual(0.0, z.imag)
        for base in [0.5, 2.0, 10.0]:
            for v in positive:
                z = cmath.log(v, base)
                self.rAssertAlmostEqual(math.log(v, base), z.real)
                self.assertEqual(0.0, z.imag)

    @requires_IEEE_754
    def test_specific_values(self):
        if False:
            return 10
        SKIP_ON_TIGER = {'tan0064'}
        osx_version = None
        if sys.platform == 'darwin':
            version_txt = platform.mac_ver()[0]
            try:
                osx_version = tuple(map(int, version_txt.split('.')))
            except ValueError:
                pass

        def rect_complex(z):
            if False:
                print('Hello World!')
            'Wrapped version of rect that accepts a complex number instead of\n            two float arguments.'
            return cmath.rect(z.real, z.imag)

        def polar_complex(z):
            if False:
                i = 10
                return i + 15
            'Wrapped version of polar that returns a complex number instead of\n            two floats.'
            return complex(*polar(z))
        for (id, fn, ar, ai, er, ei, flags) in parse_testfile(test_file):
            arg = complex(ar, ai)
            expected = complex(er, ei)
            if osx_version is not None and osx_version < (10, 5):
                if id in SKIP_ON_TIGER:
                    continue
            if fn == 'rect':
                function = rect_complex
            elif fn == 'polar':
                function = polar_complex
            else:
                function = getattr(cmath, fn)
            if 'divide-by-zero' in flags or 'invalid' in flags:
                try:
                    actual = function(arg)
                except ValueError:
                    continue
                else:
                    self.fail('ValueError not raised in test {}: {}(complex({!r}, {!r}))'.format(id, fn, ar, ai))
            if 'overflow' in flags:
                try:
                    actual = function(arg)
                except OverflowError:
                    continue
                else:
                    self.fail('OverflowError not raised in test {}: {}(complex({!r}, {!r}))'.format(id, fn, ar, ai))
            actual = function(arg)
            if 'ignore-real-sign' in flags:
                actual = complex(abs(actual.real), actual.imag)
                expected = complex(abs(expected.real), expected.imag)
            if 'ignore-imag-sign' in flags:
                actual = complex(actual.real, abs(actual.imag))
                expected = complex(expected.real, abs(expected.imag))
            if fn in ('log', 'log10'):
                real_abs_err = 2e-15
            else:
                real_abs_err = 5e-323
            error_message = '{}: {}(complex({!r}, {!r}))\nExpected: complex({!r}, {!r})\nReceived: complex({!r}, {!r})\nReceived value insufficiently close to expected value.'.format(id, fn, ar, ai, expected.real, expected.imag, actual.real, actual.imag)
            self.rAssertAlmostEqual(expected.real, actual.real, abs_err=real_abs_err, msg=error_message)
            self.rAssertAlmostEqual(expected.imag, actual.imag, msg=error_message)

    def check_polar(self, func):
        if False:
            while True:
                i = 10

        def check(arg, expected):
            if False:
                print('Hello World!')
            got = func(arg)
            for (e, g) in zip(expected, got):
                self.rAssertAlmostEqual(e, g)
        check(0, (0.0, 0.0))
        check(1, (1.0, 0.0))
        check(-1, (1.0, pi))
        check(1j, (1.0, pi / 2))
        check(-3j, (3.0, -pi / 2))
        inf = float('inf')
        check(complex(inf, 0), (inf, 0.0))
        check(complex(-inf, 0), (inf, pi))
        check(complex(3, inf), (inf, pi / 2))
        check(complex(5, -inf), (inf, -pi / 2))
        check(complex(inf, inf), (inf, pi / 4))
        check(complex(inf, -inf), (inf, -pi / 4))
        check(complex(-inf, inf), (inf, 3 * pi / 4))
        check(complex(-inf, -inf), (inf, -3 * pi / 4))
        nan = float('nan')
        check(complex(nan, 0), (nan, nan))
        check(complex(0, nan), (nan, nan))
        check(complex(nan, nan), (nan, nan))
        check(complex(inf, nan), (inf, nan))
        check(complex(-inf, nan), (inf, nan))
        check(complex(nan, inf), (inf, nan))
        check(complex(nan, -inf), (inf, nan))

    def test_polar(self):
        if False:
            while True:
                i = 10
        self.check_polar(polar)

    @cpython_only
    def test_polar_errno(self):
        if False:
            return 10
        from _testcapi import set_errno

        def polar_with_errno_set(z):
            if False:
                i = 10
                return i + 15
            set_errno(11)
            try:
                return polar(z)
            finally:
                set_errno(0)
        self.check_polar(polar_with_errno_set)

    def test_phase(self):
        if False:
            print('Hello World!')
        self.assertAlmostEqual(phase(0), 0.0)
        self.assertAlmostEqual(phase(1.0), 0.0)
        self.assertAlmostEqual(phase(-1.0), pi)
        self.assertAlmostEqual(phase(-1.0 + 1e-300j), pi)
        self.assertAlmostEqual(phase(-1.0 - 1e-300j), -pi)
        self.assertAlmostEqual(phase(1j), pi / 2)
        self.assertAlmostEqual(phase(-1j), -pi / 2)
        self.assertEqual(phase(complex(0.0, 0.0)), 0.0)
        self.assertEqual(phase(complex(0.0, -0.0)), -0.0)
        self.assertEqual(phase(complex(-0.0, 0.0)), pi)
        self.assertEqual(phase(complex(-0.0, -0.0)), -pi)
        self.assertAlmostEqual(phase(complex(-INF, -0.0)), -pi)
        self.assertAlmostEqual(phase(complex(-INF, -2.3)), -pi)
        self.assertAlmostEqual(phase(complex(-INF, -INF)), -0.75 * pi)
        self.assertAlmostEqual(phase(complex(-2.3, -INF)), -pi / 2)
        self.assertAlmostEqual(phase(complex(-0.0, -INF)), -pi / 2)
        self.assertAlmostEqual(phase(complex(0.0, -INF)), -pi / 2)
        self.assertAlmostEqual(phase(complex(2.3, -INF)), -pi / 2)
        self.assertAlmostEqual(phase(complex(INF, -INF)), -pi / 4)
        self.assertEqual(phase(complex(INF, -2.3)), -0.0)
        self.assertEqual(phase(complex(INF, -0.0)), -0.0)
        self.assertEqual(phase(complex(INF, 0.0)), 0.0)
        self.assertEqual(phase(complex(INF, 2.3)), 0.0)
        self.assertAlmostEqual(phase(complex(INF, INF)), pi / 4)
        self.assertAlmostEqual(phase(complex(2.3, INF)), pi / 2)
        self.assertAlmostEqual(phase(complex(0.0, INF)), pi / 2)
        self.assertAlmostEqual(phase(complex(-0.0, INF)), pi / 2)
        self.assertAlmostEqual(phase(complex(-2.3, INF)), pi / 2)
        self.assertAlmostEqual(phase(complex(-INF, INF)), 0.75 * pi)
        self.assertAlmostEqual(phase(complex(-INF, 2.3)), pi)
        self.assertAlmostEqual(phase(complex(-INF, 0.0)), pi)
        for z in complex_nans:
            self.assertTrue(math.isnan(phase(z)))

    def test_abs(self):
        if False:
            for i in range(10):
                print('nop')
        for z in complex_zeros:
            self.assertEqual(abs(z), 0.0)
        for z in complex_infinities:
            self.assertEqual(abs(z), INF)
        self.assertEqual(abs(complex(NAN, -INF)), INF)
        self.assertTrue(math.isnan(abs(complex(NAN, -2.3))))
        self.assertTrue(math.isnan(abs(complex(NAN, -0.0))))
        self.assertTrue(math.isnan(abs(complex(NAN, 0.0))))
        self.assertTrue(math.isnan(abs(complex(NAN, 2.3))))
        self.assertEqual(abs(complex(NAN, INF)), INF)
        self.assertEqual(abs(complex(-INF, NAN)), INF)
        self.assertTrue(math.isnan(abs(complex(-2.3, NAN))))
        self.assertTrue(math.isnan(abs(complex(-0.0, NAN))))
        self.assertTrue(math.isnan(abs(complex(0.0, NAN))))
        self.assertTrue(math.isnan(abs(complex(2.3, NAN))))
        self.assertEqual(abs(complex(INF, NAN)), INF)
        self.assertTrue(math.isnan(abs(complex(NAN, NAN))))

    @requires_IEEE_754
    def test_abs_overflows(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(OverflowError, abs, complex(1.4e+308, 1.4e+308))

    def assertCEqual(self, a, b):
        if False:
            i = 10
            return i + 15
        eps = 1e-07
        if abs(a.real - b[0]) > eps or abs(a.imag - b[1]) > eps:
            self.fail((a, b))

    def test_rect(self):
        if False:
            print('Hello World!')
        self.assertCEqual(rect(0, 0), (0, 0))
        self.assertCEqual(rect(1, 0), (1.0, 0))
        self.assertCEqual(rect(1, -pi), (-1.0, 0))
        self.assertCEqual(rect(1, pi / 2), (0, 1.0))
        self.assertCEqual(rect(1, -pi / 2), (0, -1.0))

    def test_isfinite(self):
        if False:
            while True:
                i = 10
        real_vals = [float('-inf'), -2.3, -0.0, 0.0, 2.3, float('inf'), float('nan')]
        for x in real_vals:
            for y in real_vals:
                z = complex(x, y)
                self.assertEqual(cmath.isfinite(z), math.isfinite(x) and math.isfinite(y))

    def test_isnan(self):
        if False:
            return 10
        self.assertFalse(cmath.isnan(1))
        self.assertFalse(cmath.isnan(1j))
        self.assertFalse(cmath.isnan(INF))
        self.assertTrue(cmath.isnan(NAN))
        self.assertTrue(cmath.isnan(complex(NAN, 0)))
        self.assertTrue(cmath.isnan(complex(0, NAN)))
        self.assertTrue(cmath.isnan(complex(NAN, NAN)))
        self.assertTrue(cmath.isnan(complex(NAN, INF)))
        self.assertTrue(cmath.isnan(complex(INF, NAN)))

    def test_isinf(self):
        if False:
            return 10
        self.assertFalse(cmath.isinf(1))
        self.assertFalse(cmath.isinf(1j))
        self.assertFalse(cmath.isinf(NAN))
        self.assertTrue(cmath.isinf(INF))
        self.assertTrue(cmath.isinf(complex(INF, 0)))
        self.assertTrue(cmath.isinf(complex(0, INF)))
        self.assertTrue(cmath.isinf(complex(INF, INF)))
        self.assertTrue(cmath.isinf(complex(NAN, INF)))
        self.assertTrue(cmath.isinf(complex(INF, NAN)))

    @requires_IEEE_754
    def testTanhSign(self):
        if False:
            for i in range(10):
                print('nop')
        for z in complex_zeros:
            self.assertComplexIdentical(cmath.tanh(z), z)

    @requires_IEEE_754
    def testAtanSign(self):
        if False:
            i = 10
            return i + 15
        for z in complex_zeros:
            self.assertComplexIdentical(cmath.atan(z), z)

    @requires_IEEE_754
    def testAtanhSign(self):
        if False:
            for i in range(10):
                print('nop')
        for z in complex_zeros:
            self.assertComplexIdentical(cmath.atanh(z), z)

class IsCloseTests(test_math.IsCloseTests):
    isclose = cmath.isclose

    def test_reject_complex_tolerances(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            self.isclose(1j, 1j, rel_tol=1j)
        with self.assertRaises(TypeError):
            self.isclose(1j, 1j, abs_tol=1j)
        with self.assertRaises(TypeError):
            self.isclose(1j, 1j, rel_tol=1j, abs_tol=1j)

    def test_complex_values(self):
        if False:
            print('Hello World!')
        complex_examples = [(1.0 + 1j, 1.000000000001 + 1j), (1.0 + 1j, 1.0 + 1.000000000001j), (-1.0 + 1j, -1.000000000001 + 1j), (1.0 - 1j, 1.0 - 0.999999999999j)]
        self.assertAllClose(complex_examples, rel_tol=1e-12)
        self.assertAllNotClose(complex_examples, rel_tol=1e-13)

    def test_complex_near_zero(self):
        if False:
            print('Hello World!')
        near_zero_examples = [(0.001j, 0), (0.001, 0), (0.001 + 0.001j, 0), (-0.001 + 0.001j, 0), (0.001 - 0.001j, 0), (-0.001 - 0.001j, 0)]
        self.assertAllClose(near_zero_examples, abs_tol=0.0015)
        self.assertAllNotClose(near_zero_examples, abs_tol=0.0005)
        self.assertIsClose(0.001 - 0.001j, 0.001 + 0.001j, abs_tol=0.002)
        self.assertIsNotClose(0.001 - 0.001j, 0.001 + 0.001j, abs_tol=0.001)
if __name__ == '__main__':
    unittest.main()