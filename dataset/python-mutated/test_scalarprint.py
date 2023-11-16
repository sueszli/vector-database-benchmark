""" Test printing of scalar types.

"""
import code
import platform
import pytest
import sys
from tempfile import TemporaryFile
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_MUSL

class TestRealScalars:

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        svals = [0.0, -0.0, 1, -1, np.inf, -np.inf, np.nan]
        styps = [np.float16, np.float32, np.float64, np.longdouble]
        wanted = [['0.0', '0.0', '0.0', '0.0'], ['-0.0', '-0.0', '-0.0', '-0.0'], ['1.0', '1.0', '1.0', '1.0'], ['-1.0', '-1.0', '-1.0', '-1.0'], ['inf', 'inf', 'inf', 'inf'], ['-inf', '-inf', '-inf', '-inf'], ['nan', 'nan', 'nan', 'nan']]
        for (wants, val) in zip(wanted, svals):
            for (want, styp) in zip(wants, styps):
                msg = 'for str({}({}))'.format(np.dtype(styp).name, repr(val))
                assert_equal(str(styp(val)), want, err_msg=msg)

    def test_scalar_cutoffs(self):
        if False:
            print('Hello World!')

        def check(v):
            if False:
                i = 10
                return i + 15
            assert_equal(str(np.float64(v)), str(v))
            assert_equal(str(np.float64(v)), repr(v))
            assert_equal(repr(np.float64(v)), f'np.float64({v!r})')
            assert_equal(repr(np.float64(v)), f'np.float64({v})')
        check(1.1234567890123457)
        check(0.011234567890123457)
        check(1e-05)
        check(0.0001)
        check(1000000000000000.0)
        check(1e+16)

    def test_py2_float_print(self):
        if False:
            i = 10
            return i + 15
        x = np.double(0.1999999999999)
        with TemporaryFile('r+t') as f:
            print(x, file=f)
            f.seek(0)
            output = f.read()
        assert_equal(output, str(x) + '\n')

        def userinput():
            if False:
                print('Hello World!')
            yield 'np.sqrt(2)'
            raise EOFError
        gen = userinput()
        input_func = lambda prompt='': next(gen)
        with TemporaryFile('r+t') as fo, TemporaryFile('r+t') as fe:
            (orig_stdout, orig_stderr) = (sys.stdout, sys.stderr)
            (sys.stdout, sys.stderr) = (fo, fe)
            code.interact(local={'np': np}, readfunc=input_func, banner='')
            (sys.stdout, sys.stderr) = (orig_stdout, orig_stderr)
            fo.seek(0)
            capture = fo.read().strip()
        assert_equal(capture, repr(np.sqrt(2)))

    def test_dragon4(self):
        if False:
            return 10
        fpos32 = lambda x, **k: np.format_float_positional(np.float32(x), **k)
        fsci32 = lambda x, **k: np.format_float_scientific(np.float32(x), **k)
        fpos64 = lambda x, **k: np.format_float_positional(np.float64(x), **k)
        fsci64 = lambda x, **k: np.format_float_scientific(np.float64(x), **k)
        preckwd = lambda prec: {'unique': False, 'precision': prec}
        assert_equal(fpos32('1.0'), '1.')
        assert_equal(fsci32('1.0'), '1.e+00')
        assert_equal(fpos32('10.234'), '10.234')
        assert_equal(fpos32('-10.234'), '-10.234')
        assert_equal(fsci32('10.234'), '1.0234e+01')
        assert_equal(fsci32('-10.234'), '-1.0234e+01')
        assert_equal(fpos32('1000.0'), '1000.')
        assert_equal(fpos32('1.0', precision=0), '1.')
        assert_equal(fsci32('1.0', precision=0), '1.e+00')
        assert_equal(fpos32('10.234', precision=0), '10.')
        assert_equal(fpos32('-10.234', precision=0), '-10.')
        assert_equal(fsci32('10.234', precision=0), '1.e+01')
        assert_equal(fsci32('-10.234', precision=0), '-1.e+01')
        assert_equal(fpos32('10.234', precision=2), '10.23')
        assert_equal(fsci32('-10.234', precision=2), '-1.02e+01')
        assert_equal(fsci64('9.9999999999999995e-08', **preckwd(16)), '9.9999999999999995e-08')
        assert_equal(fsci64('9.8813129168249309e-324', **preckwd(16)), '9.8813129168249309e-324')
        assert_equal(fsci64('9.9999999999999694e-311', **preckwd(16)), '9.9999999999999694e-311')
        assert_equal(fpos32('3.14159265358979323846', **preckwd(10)), '3.1415927410')
        assert_equal(fsci32('3.14159265358979323846', **preckwd(10)), '3.1415927410e+00')
        assert_equal(fpos64('3.14159265358979323846', **preckwd(10)), '3.1415926536')
        assert_equal(fsci64('3.14159265358979323846', **preckwd(10)), '3.1415926536e+00')
        assert_equal(fpos32('299792458.0', **preckwd(5)), '299792448.00000')
        assert_equal(fsci32('299792458.0', **preckwd(5)), '2.99792e+08')
        assert_equal(fpos64('299792458.0', **preckwd(5)), '299792458.00000')
        assert_equal(fsci64('299792458.0', **preckwd(5)), '2.99792e+08')
        assert_equal(fpos32('3.14159265358979323846', **preckwd(25)), '3.1415927410125732421875000')
        assert_equal(fpos64('3.14159265358979323846', **preckwd(50)), '3.14159265358979311599796346854418516159057617187500')
        assert_equal(fpos64('3.14159265358979323846'), '3.141592653589793')
        assert_equal(fpos32(0.5 ** (126 + 23), unique=False, precision=149), '0.00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125')
        assert_equal(fpos64(5e-324, unique=False, precision=1074), '0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004940656458412465441765687928682213723650598026143247644255856825006755072702087518652998363616359923797965646954457177309266567103559397963987747960107818781263007131903114045278458171678489821036887186360569987307230500063874091535649843873124733972731696151400317153853980741262385655911710266585566867681870395603106249319452715914924553293054565444011274801297099995419319894090804165633245247571478690147267801593552386115501348035264934720193790268107107491703332226844753335720832431936092382893458368060106011506169809753078342277318329247904982524730776375927247874656084778203734469699533647017972677717585125660551199131504891101451037862738167250955837389733598993664809941164205702637090279242767544565229087538682506419718265533447265625')
        f32x = np.finfo(np.float32).max
        assert_equal(fpos32(f32x, **preckwd(0)), '340282346638528859811704183484516925440.')
        assert_equal(fpos64(np.finfo(np.float64).max, **preckwd(0)), '179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.')
        assert_equal(fpos32(f32x), '340282350000000000000000000000000000000.')
        assert_equal(fpos32(f32x, unique=True, fractional=True, precision=0), '340282350000000000000000000000000000000.')
        assert_equal(fpos32(f32x, unique=True, fractional=True, precision=4), '340282350000000000000000000000000000000.')
        assert_equal(fpos32(f32x, unique=True, fractional=True, min_digits=0), '340282346638528859811704183484516925440.')
        assert_equal(fpos32(f32x, unique=True, fractional=True, min_digits=4), '340282346638528859811704183484516925440.0000')
        assert_equal(fpos32(f32x, unique=True, fractional=True, min_digits=4, precision=4), '340282346638528859811704183484516925440.0000')
        assert_raises(ValueError, fpos32, f32x, unique=True, fractional=False, precision=0)
        assert_equal(fpos32(f32x, unique=True, fractional=False, precision=4), '340300000000000000000000000000000000000.')
        assert_equal(fpos32(f32x, unique=True, fractional=False, precision=20), '340282350000000000000000000000000000000.')
        assert_equal(fpos32(f32x, unique=True, fractional=False, min_digits=4), '340282350000000000000000000000000000000.')
        assert_equal(fpos32(f32x, unique=True, fractional=False, min_digits=20), '340282346638528859810000000000000000000.')
        assert_equal(fpos32(f32x, unique=True, fractional=False, min_digits=15), '340282346638529000000000000000000000000.')
        assert_equal(fpos32(f32x, unique=False, fractional=False, precision=4), '340300000000000000000000000000000000000.')
        a = np.float64.fromhex('-1p-97')
        assert_equal(fsci64(a, unique=True), '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=False, precision=15), '-6.310887241768094e-30')
        assert_equal(fsci64(a, unique=True, precision=15), '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, min_digits=15), '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, precision=15, min_digits=15), '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, precision=14), '-6.31088724176809e-30')
        assert_equal(fsci64(a, unique=True, min_digits=16), '-6.3108872417680944e-30')
        assert_equal(fsci64(a, unique=True, precision=16), '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, min_digits=14), '-6.310887241768095e-30')
        assert_equal(fsci64('1e120', min_digits=3), '1.000e+120')
        assert_equal(fsci64('1e100', min_digits=3), '1.000e+100')
        assert_equal(fpos32('1.0', unique=False, precision=3), '1.000')
        assert_equal(fpos64('1.0', unique=False, precision=3), '1.000')
        assert_equal(fsci32('1.0', unique=False, precision=3), '1.000e+00')
        assert_equal(fsci64('1.0', unique=False, precision=3), '1.000e+00')
        assert_equal(fpos32('1.5', unique=False, precision=3), '1.500')
        assert_equal(fpos64('1.5', unique=False, precision=3), '1.500')
        assert_equal(fsci32('1.5', unique=False, precision=3), '1.500e+00')
        assert_equal(fsci64('1.5', unique=False, precision=3), '1.500e+00')
        assert_equal(fpos64('324', unique=False, precision=5, fractional=False), '324.00')

    def test_dragon4_interface(self):
        if False:
            print('Hello World!')
        tps = [np.float16, np.float32, np.float64]
        if hasattr(np, 'float128') and (not IS_MUSL):
            tps.append(np.float128)
        fpos = np.format_float_positional
        fsci = np.format_float_scientific
        for tp in tps:
            assert_equal(fpos(tp('1.0'), pad_left=4, pad_right=4), '   1.    ')
            assert_equal(fpos(tp('-1.0'), pad_left=4, pad_right=4), '  -1.    ')
            assert_equal(fpos(tp('-10.2'), pad_left=4, pad_right=4), ' -10.2   ')
            assert_equal(fsci(tp('1.23e1'), exp_digits=5), '1.23e+00001')
            assert_equal(fpos(tp('1.0'), unique=False, precision=4), '1.0000')
            assert_equal(fsci(tp('1.0'), unique=False, precision=4), '1.0000e+00')
            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='k'), '1.0000')
            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='.'), '1.')
            assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='.'), '1.2' if tp != np.float16 else '1.2002')
            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='0'), '1.0')
            assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='0'), '1.2' if tp != np.float16 else '1.2002')
            assert_equal(fpos(tp('1.'), trim='0'), '1.0')
            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='-'), '1')
            assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='-'), '1.2' if tp != np.float16 else '1.2002')
            assert_equal(fpos(tp('1.'), trim='-'), '1')
            assert_equal(fpos(tp('1.001'), precision=1, trim='-'), '1')

    @pytest.mark.skipif(not platform.machine().startswith('ppc64'), reason='only applies to ppc float128 values')
    def test_ppc64_ibm_double_double128(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.float128('2.123123123123123123123123123123123e-286')
        got = [str(x / np.float128('2e' + str(i))) for i in range(0, 40)]
        expected = ['1.06156156156156156156156156156157e-286', '1.06156156156156156156156156156158e-287', '1.06156156156156156156156156156159e-288', '1.0615615615615615615615615615616e-289', '1.06156156156156156156156156156157e-290', '1.06156156156156156156156156156156e-291', '1.0615615615615615615615615615616e-292', '1.0615615615615615615615615615615e-293', '1.061561561561561561561561561562e-294', '1.06156156156156156156156156155e-295', '1.0615615615615615615615615616e-296', '1.06156156156156156156156156e-297', '1.06156156156156156156156157e-298', '1.0615615615615615615615616e-299', '1.06156156156156156156156e-300', '1.06156156156156156156155e-301', '1.0615615615615615615616e-302', '1.061561561561561561562e-303', '1.06156156156156156156e-304', '1.0615615615615615618e-305', '1.06156156156156156e-306', '1.06156156156156157e-307', '1.0615615615615616e-308', '1.06156156156156e-309', '1.06156156156157e-310', '1.0615615615616e-311', '1.06156156156e-312', '1.06156156154e-313', '1.0615615616e-314', '1.06156156e-315', '1.06156155e-316', '1.061562e-317', '1.06156e-318', '1.06155e-319', '1.0617e-320', '1.06e-321', '1.04e-322', '1e-323', '0.0', '0.0']
        assert_equal(got, expected)
        a = np.float128('2') / np.float128('3')
        b = np.float128(str(a))
        assert_equal(str(a), str(b))
        assert_(a != b)

    def float32_roundtrip(self):
        if False:
            while True:
                i = 10
        x = np.float32(1024 - 2 ** (-14))
        y = np.float32(1024 - 2 ** (-13))
        assert_(repr(x) != repr(y))
        assert_equal(np.float32(repr(x)), x)
        assert_equal(np.float32(repr(y)), y)

    def float64_vs_python(self):
        if False:
            return 10
        assert_equal(repr(np.float64(0.1)), repr(0.1))
        assert_(repr(np.float64(0.20000000000000004)) != repr(0.2))