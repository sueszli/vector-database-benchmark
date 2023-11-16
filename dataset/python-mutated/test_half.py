import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM

def assert_raises_fpe(strmatch, callable, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    try:
        callable(*args, **kwargs)
    except FloatingPointError as exc:
        assert_(str(exc).find(strmatch) >= 0, 'Did not raise floating point %s error' % strmatch)
    else:
        assert_(False, 'Did not raise floating point %s error' % strmatch)

class TestHalf:

    def setup_method(self):
        if False:
            return 10
        self.all_f16 = np.arange(65536, dtype=uint16)
        self.all_f16.dtype = float16
        with np.errstate(invalid='ignore'):
            self.all_f32 = np.array(self.all_f16, dtype=float32)
            self.all_f64 = np.array(self.all_f16, dtype=float64)
        self.nonan_f16 = np.concatenate((np.arange(64512, 32767, -1, dtype=uint16), np.arange(0, 31745, 1, dtype=uint16)))
        self.nonan_f16.dtype = float16
        self.nonan_f32 = np.array(self.nonan_f16, dtype=float32)
        self.nonan_f64 = np.array(self.nonan_f16, dtype=float64)
        self.finite_f16 = self.nonan_f16[1:-1]
        self.finite_f32 = self.nonan_f32[1:-1]
        self.finite_f64 = self.nonan_f64[1:-1]

    def test_half_conversions(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks that all 16-bit values survive conversion\n           to/from 32-bit and 64-bit float'
        with np.errstate(invalid='ignore'):
            b = np.array(self.all_f32, dtype=float16)
        b_nn = b == b
        assert_equal(self.all_f16[b_nn].view(dtype=uint16), b[b_nn].view(dtype=uint16))
        with np.errstate(invalid='ignore'):
            b = np.array(self.all_f64, dtype=float16)
        b_nn = b == b
        assert_equal(self.all_f16[b_nn].view(dtype=uint16), b[b_nn].view(dtype=uint16))
        a_ld = np.array(self.nonan_f16, dtype=np.longdouble)
        b = np.array(a_ld, dtype=float16)
        assert_equal(self.nonan_f16.view(dtype=uint16), b.view(dtype=uint16))
        i_int = np.arange(-2048, 2049)
        i_f16 = np.array(i_int, dtype=float16)
        j = np.array(i_f16, dtype=int)
        assert_equal(i_int, j)

    @pytest.mark.parametrize('string_dt', ['S', 'U'])
    def test_half_conversion_to_string(self, string_dt):
        if False:
            while True:
                i = 10
        expected_dt = np.dtype(f'{string_dt}32')
        assert np.promote_types(np.float16, string_dt) == expected_dt
        assert np.promote_types(string_dt, np.float16) == expected_dt
        arr = np.ones(3, dtype=np.float16).astype(string_dt)
        assert arr.dtype == expected_dt

    @pytest.mark.parametrize('string_dt', ['S', 'U'])
    def test_half_conversion_from_string(self, string_dt):
        if False:
            print('Hello World!')
        string = np.array('3.1416', dtype=string_dt)
        assert string.astype(np.float16) == np.array(3.1416, dtype=np.float16)

    @pytest.mark.parametrize('offset', [None, 'up', 'down'])
    @pytest.mark.parametrize('shift', [None, 'up', 'down'])
    @pytest.mark.parametrize('float_t', [np.float32, np.float64])
    @np._no_nep50_warning()
    def test_half_conversion_rounding(self, float_t, shift, offset):
        if False:
            return 10
        max_pattern = np.float16(np.finfo(np.float16).max).view(np.uint16)
        f16s_patterns = np.arange(0, max_pattern + 1, dtype=np.uint16)
        f16s_float = f16s_patterns.view(np.float16).astype(float_t)
        if shift == 'up':
            f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[1:]
        elif shift == 'down':
            f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[:-1]
        else:
            f16s_float = f16s_float[1:-1]
        if offset == 'up':
            f16s_float = np.nextafter(f16s_float, float_t(np.inf))
        elif offset == 'down':
            f16s_float = np.nextafter(f16s_float, float_t(-np.inf))
        res_patterns = f16s_float.astype(np.float16).view(np.uint16)
        cmp_patterns = f16s_patterns[1:-1].copy()
        if shift == 'down' and offset != 'up':
            shift_pattern = -1
        elif shift == 'up' and offset != 'down':
            shift_pattern = 1
        else:
            shift_pattern = 0
        if offset is None:
            cmp_patterns[0::2].view(np.int16)[...] += shift_pattern
        else:
            cmp_patterns.view(np.int16)[...] += shift_pattern
        assert_equal(res_patterns, cmp_patterns)

    @pytest.mark.parametrize(['float_t', 'uint_t', 'bits'], [(np.float32, np.uint32, 23), (np.float64, np.uint64, 52)])
    def test_half_conversion_denormal_round_even(self, float_t, uint_t, bits):
        if False:
            for i in range(10):
                print('nop')
        smallest_value = np.uint16(1).view(np.float16).astype(float_t)
        assert smallest_value == 2 ** (-24)
        rounded_to_zero = smallest_value / float_t(2)
        assert rounded_to_zero.astype(np.float16) == 0
        for i in range(bits):
            larger_pattern = rounded_to_zero.view(uint_t) | uint_t(1 << i)
            larger_value = larger_pattern.view(float_t)
            assert larger_value.astype(np.float16) == smallest_value

    def test_nans_infs(self):
        if False:
            for i in range(10):
                print('nop')
        with np.errstate(all='ignore'):
            assert_equal(np.isnan(self.all_f16), np.isnan(self.all_f32))
            assert_equal(np.isinf(self.all_f16), np.isinf(self.all_f32))
            assert_equal(np.isfinite(self.all_f16), np.isfinite(self.all_f32))
            assert_equal(np.signbit(self.all_f16), np.signbit(self.all_f32))
            assert_equal(np.spacing(float16(65504)), np.inf)
            nan = float16(np.nan)
            assert_(not (self.all_f16 == nan).any())
            assert_(not (nan == self.all_f16).any())
            assert_((self.all_f16 != nan).all())
            assert_((nan != self.all_f16).all())
            assert_(not (self.all_f16 < nan).any())
            assert_(not (nan < self.all_f16).any())
            assert_(not (self.all_f16 <= nan).any())
            assert_(not (nan <= self.all_f16).any())
            assert_(not (self.all_f16 > nan).any())
            assert_(not (nan > self.all_f16).any())
            assert_(not (self.all_f16 >= nan).any())
            assert_(not (nan >= self.all_f16).any())

    def test_half_values(self):
        if False:
            return 10
        'Confirms a small number of known half values'
        a = np.array([1.0, -1.0, 2.0, -2.0, 0.0999755859375, 0.333251953125, 65504, -65504, 2.0 ** (-14), -2.0 ** (-14), 2.0 ** (-24), -2.0 ** (-24), 0, -1 / 1e309, np.inf, -np.inf])
        b = np.array([15360, 48128, 16384, 49152, 11878, 13653, 31743, 64511, 1024, 33792, 1, 32769, 0, 32768, 31744, 64512], dtype=uint16)
        b.dtype = float16
        assert_equal(a, b)

    def test_half_rounding(self):
        if False:
            return 10
        'Checks that rounding when converting to half is correct'
        a = np.array([2.0 ** (-25) + 2.0 ** (-35), 2.0 ** (-25), 2.0 ** (-26), 1.0 + 2.0 ** (-11) + 2.0 ** (-16), 1.0 + 2.0 ** (-11), 1.0 + 2.0 ** (-12), 65519, 65520], dtype=float64)
        rounded = [2.0 ** (-24), 0.0, 0.0, 1.0 + 2.0 ** (-10), 1.0, 1.0, 65504, np.inf]
        with np.errstate(over='ignore'):
            b = np.array(a, dtype=float16)
        assert_equal(b, rounded)
        a = np.array(a, dtype=float32)
        with np.errstate(over='ignore'):
            b = np.array(a, dtype=float16)
        assert_equal(b, rounded)

    def test_half_correctness(self):
        if False:
            for i in range(10):
                print('nop')
        'Take every finite float16, and check the casting functions with\n           a manual conversion.'
        a_bits = self.finite_f16.view(dtype=uint16)
        a_sgn = (-1.0) ** ((a_bits & 32768) >> 15)
        a_exp = np.array((a_bits & 31744) >> 10, dtype=np.int32) - 15
        a_man = (a_bits & 1023) * 2.0 ** (-10)
        a_man[a_exp != -15] += 1
        a_exp[a_exp == -15] = -14
        a_manual = a_sgn * a_man * 2.0 ** a_exp
        a32_fail = np.nonzero(self.finite_f32 != a_manual)[0]
        if len(a32_fail) != 0:
            bad_index = a32_fail[0]
            assert_equal(self.finite_f32, a_manual, 'First non-equal is half value 0x%x -> %g != %g' % (a_bits[bad_index], self.finite_f32[bad_index], a_manual[bad_index]))
        a64_fail = np.nonzero(self.finite_f64 != a_manual)[0]
        if len(a64_fail) != 0:
            bad_index = a64_fail[0]
            assert_equal(self.finite_f64, a_manual, 'First non-equal is half value 0x%x -> %g != %g' % (a_bits[bad_index], self.finite_f64[bad_index], a_manual[bad_index]))

    def test_half_ordering(self):
        if False:
            while True:
                i = 10
        'Make sure comparisons are working right'
        a = self.nonan_f16[::-1].copy()
        b = np.array(a, dtype=float32)
        a.sort()
        b.sort()
        assert_equal(a, b)
        assert_((a[:-1] <= a[1:]).all())
        assert_(not (a[:-1] > a[1:]).any())
        assert_((a[1:] >= a[:-1]).all())
        assert_(not (a[1:] < a[:-1]).any())
        assert_equal(np.nonzero(a[:-1] < a[1:])[0].size, a.size - 2)
        assert_equal(np.nonzero(a[1:] > a[:-1])[0].size, a.size - 2)

    def test_half_funcs(self):
        if False:
            i = 10
            return i + 15
        'Test the various ArrFuncs'
        assert_equal(np.arange(10, dtype=float16), np.arange(10, dtype=float32))
        a = np.zeros((5,), dtype=float16)
        a.fill(1)
        assert_equal(a, np.ones((5,), dtype=float16))
        a = np.array([0, 0, -1, -1 / 1e+20, 0, 2.0 ** (-24), 7.629e-06], dtype=float16)
        assert_equal(a.nonzero()[0], [2, 5, 6])
        a = a.byteswap()
        a = a.view(a.dtype.newbyteorder())
        assert_equal(a.nonzero()[0], [2, 5, 6])
        a = np.arange(0, 10, 0.5, dtype=float16)
        b = np.ones((20,), dtype=float16)
        assert_equal(np.dot(a, b), 95)
        a = np.array([0, -np.inf, -2, 0.5, 12.55, 7.3, 2.1, 12.4], dtype=float16)
        assert_equal(a.argmax(), 4)
        a = np.array([0, -np.inf, -2, np.inf, 12.55, np.nan, 2.1, 12.4], dtype=float16)
        assert_equal(a.argmax(), 5)
        a = np.arange(10, dtype=float16)
        for i in range(10):
            assert_equal(a.item(i), i)

    def test_spacing_nextafter(self):
        if False:
            return 10
        'Test np.spacing and np.nextafter'
        a = np.arange(31744, dtype=uint16)
        hinf = np.array((np.inf,), dtype=float16)
        hnan = np.array((np.nan,), dtype=float16)
        a_f16 = a.view(dtype=float16)
        assert_equal(np.spacing(a_f16[:-1]), a_f16[1:] - a_f16[:-1])
        assert_equal(np.nextafter(a_f16[:-1], hinf), a_f16[1:])
        assert_equal(np.nextafter(a_f16[0], -hinf), -a_f16[1])
        assert_equal(np.nextafter(a_f16[1:], -hinf), a_f16[:-1])
        assert_equal(np.nextafter(hinf, a_f16), a_f16[-1])
        assert_equal(np.nextafter(-hinf, a_f16), -a_f16[-1])
        assert_equal(np.nextafter(hinf, hinf), hinf)
        assert_equal(np.nextafter(hinf, -hinf), a_f16[-1])
        assert_equal(np.nextafter(-hinf, hinf), -a_f16[-1])
        assert_equal(np.nextafter(-hinf, -hinf), -hinf)
        assert_equal(np.nextafter(a_f16, hnan), hnan[0])
        assert_equal(np.nextafter(hnan, a_f16), hnan[0])
        assert_equal(np.nextafter(hnan, hnan), hnan)
        assert_equal(np.nextafter(hinf, hnan), hnan)
        assert_equal(np.nextafter(hnan, hinf), hnan)
        a |= 32768
        assert_equal(np.spacing(a_f16[0]), np.spacing(a_f16[1]))
        assert_equal(np.spacing(a_f16[1:]), a_f16[:-1] - a_f16[1:])
        assert_equal(np.nextafter(a_f16[0], hinf), -a_f16[1])
        assert_equal(np.nextafter(a_f16[1:], hinf), a_f16[:-1])
        assert_equal(np.nextafter(a_f16[:-1], -hinf), a_f16[1:])
        assert_equal(np.nextafter(hinf, a_f16), -a_f16[-1])
        assert_equal(np.nextafter(-hinf, a_f16), a_f16[-1])
        assert_equal(np.nextafter(a_f16, hnan), hnan[0])
        assert_equal(np.nextafter(hnan, a_f16), hnan[0])

    def test_half_ufuncs(self):
        if False:
            return 10
        'Test the various ufuncs'
        a = np.array([0, 1, 2, 4, 2], dtype=float16)
        b = np.array([-2, 5, 1, 4, 3], dtype=float16)
        c = np.array([0, -1, -np.inf, np.nan, 6], dtype=float16)
        assert_equal(np.add(a, b), [-2, 6, 3, 8, 5])
        assert_equal(np.subtract(a, b), [2, -4, 1, 0, -1])
        assert_equal(np.multiply(a, b), [0, 5, 2, 16, 6])
        assert_equal(np.divide(a, b), [0, 0.199951171875, 2, 1, 0.66650390625])
        assert_equal(np.equal(a, b), [False, False, False, True, False])
        assert_equal(np.not_equal(a, b), [True, True, True, False, True])
        assert_equal(np.less(a, b), [False, True, False, False, True])
        assert_equal(np.less_equal(a, b), [False, True, False, True, True])
        assert_equal(np.greater(a, b), [True, False, True, False, False])
        assert_equal(np.greater_equal(a, b), [True, False, True, True, False])
        assert_equal(np.logical_and(a, b), [False, True, True, True, True])
        assert_equal(np.logical_or(a, b), [True, True, True, True, True])
        assert_equal(np.logical_xor(a, b), [True, False, False, False, False])
        assert_equal(np.logical_not(a), [True, False, False, False, False])
        assert_equal(np.isnan(c), [False, False, False, True, False])
        assert_equal(np.isinf(c), [False, False, True, False, False])
        assert_equal(np.isfinite(c), [True, True, False, False, True])
        assert_equal(np.signbit(b), [True, False, False, False, False])
        assert_equal(np.copysign(b, a), [2, 5, 1, 4, 3])
        assert_equal(np.maximum(a, b), [0, 5, 2, 4, 3])
        x = np.maximum(b, c)
        assert_(np.isnan(x[3]))
        x[3] = 0
        assert_equal(x, [0, 5, 1, 0, 6])
        assert_equal(np.minimum(a, b), [-2, 1, 1, 4, 2])
        x = np.minimum(b, c)
        assert_(np.isnan(x[3]))
        x[3] = 0
        assert_equal(x, [-2, -1, -np.inf, 0, 3])
        assert_equal(np.fmax(a, b), [0, 5, 2, 4, 3])
        assert_equal(np.fmax(b, c), [0, 5, 1, 4, 6])
        assert_equal(np.fmin(a, b), [-2, 1, 1, 4, 2])
        assert_equal(np.fmin(b, c), [-2, -1, -np.inf, 4, 3])
        assert_equal(np.floor_divide(a, b), [0, 0, 2, 1, 0])
        assert_equal(np.remainder(a, b), [0, 1, 0, 0, 2])
        assert_equal(np.divmod(a, b), ([0, 0, 2, 1, 0], [0, 1, 0, 0, 2]))
        assert_equal(np.square(b), [4, 25, 1, 16, 9])
        assert_equal(np.reciprocal(b), [-0.5, 0.199951171875, 1, 0.25, 0.333251953125])
        assert_equal(np.ones_like(b), [1, 1, 1, 1, 1])
        assert_equal(np.conjugate(b), b)
        assert_equal(np.absolute(b), [2, 5, 1, 4, 3])
        assert_equal(np.negative(b), [2, -5, -1, -4, -3])
        assert_equal(np.positive(b), b)
        assert_equal(np.sign(b), [-1, 1, 1, 1, 1])
        assert_equal(np.modf(b), ([0, 0, 0, 0, 0], b))
        assert_equal(np.frexp(b), ([-0.5, 0.625, 0.5, 0.5, 0.75], [2, 3, 1, 3, 2]))
        assert_equal(np.ldexp(b, [0, 1, 2, 4, 2]), [-2, 10, 4, 64, 12])

    @np._no_nep50_warning()
    def test_half_coercion(self, weak_promotion):
        if False:
            print('Hello World!')
        'Test that half gets coerced properly with the other types'
        a16 = np.array((1,), dtype=float16)
        a32 = np.array((1,), dtype=float32)
        b16 = float16(1)
        b32 = float32(1)
        assert np.power(a16, 2).dtype == float16
        assert np.power(a16, 2.0).dtype == float16
        assert np.power(a16, b16).dtype == float16
        expected_dt = float32 if weak_promotion else float16
        assert np.power(a16, b32).dtype == expected_dt
        assert np.power(a16, a16).dtype == float16
        assert np.power(a16, a32).dtype == float32
        expected_dt = float16 if weak_promotion else float64
        assert np.power(b16, 2).dtype == expected_dt
        assert np.power(b16, 2.0).dtype == expected_dt
        assert np.power(b16, b16).dtype, float16
        assert np.power(b16, b32).dtype, float32
        assert np.power(b16, a16).dtype, float16
        assert np.power(b16, a32).dtype, float32
        assert np.power(a32, a16).dtype == float32
        assert np.power(a32, b16).dtype == float32
        expected_dt = float32 if weak_promotion else float16
        assert np.power(b32, a16).dtype == expected_dt
        assert np.power(b32, b16).dtype == float32

    @pytest.mark.skipif(platform.machine() == 'armv5tel', reason='See gh-413.')
    @pytest.mark.skipif(IS_WASM, reason="fp exceptions don't work in wasm.")
    def test_half_fpe(self):
        if False:
            while True:
                i = 10
        with np.errstate(all='raise'):
            sx16 = np.array((0.0001,), dtype=float16)
            bx16 = np.array((10000.0,), dtype=float16)
            sy16 = float16(0.0001)
            by16 = float16(10000.0)
            assert_raises_fpe('underflow', lambda a, b: a * b, sx16, sx16)
            assert_raises_fpe('underflow', lambda a, b: a * b, sx16, sy16)
            assert_raises_fpe('underflow', lambda a, b: a * b, sy16, sx16)
            assert_raises_fpe('underflow', lambda a, b: a * b, sy16, sy16)
            assert_raises_fpe('underflow', lambda a, b: a / b, sx16, bx16)
            assert_raises_fpe('underflow', lambda a, b: a / b, sx16, by16)
            assert_raises_fpe('underflow', lambda a, b: a / b, sy16, bx16)
            assert_raises_fpe('underflow', lambda a, b: a / b, sy16, by16)
            assert_raises_fpe('underflow', lambda a, b: a / b, float16(2.0 ** (-14)), float16(2 ** 11))
            assert_raises_fpe('underflow', lambda a, b: a / b, float16(-2.0 ** (-14)), float16(2 ** 11))
            assert_raises_fpe('underflow', lambda a, b: a / b, float16(2.0 ** (-14) + 2 ** (-24)), float16(2))
            assert_raises_fpe('underflow', lambda a, b: a / b, float16(-2.0 ** (-14) - 2 ** (-24)), float16(2))
            assert_raises_fpe('underflow', lambda a, b: a / b, float16(2.0 ** (-14) + 2 ** (-23)), float16(4))
            assert_raises_fpe('overflow', lambda a, b: a * b, bx16, bx16)
            assert_raises_fpe('overflow', lambda a, b: a * b, bx16, by16)
            assert_raises_fpe('overflow', lambda a, b: a * b, by16, bx16)
            assert_raises_fpe('overflow', lambda a, b: a * b, by16, by16)
            assert_raises_fpe('overflow', lambda a, b: a / b, bx16, sx16)
            assert_raises_fpe('overflow', lambda a, b: a / b, bx16, sy16)
            assert_raises_fpe('overflow', lambda a, b: a / b, by16, sx16)
            assert_raises_fpe('overflow', lambda a, b: a / b, by16, sy16)
            assert_raises_fpe('overflow', lambda a, b: a + b, float16(65504), float16(17))
            assert_raises_fpe('overflow', lambda a, b: a - b, float16(-65504), float16(17))
            assert_raises_fpe('overflow', np.nextafter, float16(65504), float16(np.inf))
            assert_raises_fpe('overflow', np.nextafter, float16(-65504), float16(-np.inf))
            assert_raises_fpe('overflow', np.spacing, float16(65504))
            assert_raises_fpe('invalid', np.divide, float16(np.inf), float16(np.inf))
            assert_raises_fpe('invalid', np.spacing, float16(np.inf))
            assert_raises_fpe('invalid', np.spacing, float16(np.nan))
            float16(65472) + float16(32)
            float16(2 ** (-13)) / float16(2)
            float16(2 ** (-14)) / float16(2 ** 10)
            np.spacing(float16(-65504))
            np.nextafter(float16(65504), float16(-np.inf))
            np.nextafter(float16(-65504), float16(np.inf))
            np.nextafter(float16(np.inf), float16(0))
            np.nextafter(float16(-np.inf), float16(0))
            np.nextafter(float16(0), float16(np.nan))
            np.nextafter(float16(np.nan), float16(0))
            float16(2 ** (-14)) / float16(2 ** 10)
            float16(-2 ** (-14)) / float16(2 ** 10)
            float16(2 ** (-14) + 2 ** (-23)) / float16(2)
            float16(-2 ** (-14) - 2 ** (-23)) / float16(2)

    def test_half_array_interface(self):
        if False:
            return 10
        'Test that half is compatible with __array_interface__'

        class Dummy:
            pass
        a = np.ones((1,), dtype=float16)
        b = Dummy()
        b.__array_interface__ = a.__array_interface__
        c = np.array(b)
        assert_(c.dtype == float16)
        assert_equal(a, c)