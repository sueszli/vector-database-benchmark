import time
from datetime import date
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose, assert_raises
from numpy.lib._iotools import LineSplitter, NameValidator, StringConverter, has_nested_fields, easy_dtype, flatten_dtype

class TestLineSplitter:
    """Tests the LineSplitter class."""

    def test_no_delimiter(self):
        if False:
            print('Hello World!')
        'Test LineSplitter w/o delimiter'
        strg = ' 1 2 3 4  5 # test'
        test = LineSplitter()(strg)
        assert_equal(test, ['1', '2', '3', '4', '5'])
        test = LineSplitter('')(strg)
        assert_equal(test, ['1', '2', '3', '4', '5'])

    def test_space_delimiter(self):
        if False:
            print('Hello World!')
        'Test space delimiter'
        strg = ' 1 2 3 4  5 # test'
        test = LineSplitter(' ')(strg)
        assert_equal(test, ['1', '2', '3', '4', '', '5'])
        test = LineSplitter('  ')(strg)
        assert_equal(test, ['1 2 3 4', '5'])

    def test_tab_delimiter(self):
        if False:
            while True:
                i = 10
        'Test tab delimiter'
        strg = ' 1\t 2\t 3\t 4\t 5  6'
        test = LineSplitter('\t')(strg)
        assert_equal(test, ['1', '2', '3', '4', '5  6'])
        strg = ' 1  2\t 3  4\t 5  6'
        test = LineSplitter('\t')(strg)
        assert_equal(test, ['1  2', '3  4', '5  6'])

    def test_other_delimiter(self):
        if False:
            while True:
                i = 10
        'Test LineSplitter on delimiter'
        strg = '1,2,3,4,,5'
        test = LineSplitter(',')(strg)
        assert_equal(test, ['1', '2', '3', '4', '', '5'])
        strg = ' 1,2,3,4,,5 # test'
        test = LineSplitter(',')(strg)
        assert_equal(test, ['1', '2', '3', '4', '', '5'])
        strg = b' 1,2,3,4,,5 % test'
        test = LineSplitter(delimiter=b',', comments=b'%')(strg)
        assert_equal(test, ['1', '2', '3', '4', '', '5'])

    def test_constant_fixed_width(self):
        if False:
            return 10
        'Test LineSplitter w/ fixed-width fields'
        strg = '  1  2  3  4     5   # test'
        test = LineSplitter(3)(strg)
        assert_equal(test, ['1', '2', '3', '4', '', '5', ''])
        strg = '  1     3  4  5  6# test'
        test = LineSplitter(20)(strg)
        assert_equal(test, ['1     3  4  5  6'])
        strg = '  1     3  4  5  6# test'
        test = LineSplitter(30)(strg)
        assert_equal(test, ['1     3  4  5  6'])

    def test_variable_fixed_width(self):
        if False:
            return 10
        strg = '  1     3  4  5  6# test'
        test = LineSplitter((3, 6, 6, 3))(strg)
        assert_equal(test, ['1', '3', '4  5', '6'])
        strg = '  1     3  4  5  6# test'
        test = LineSplitter((6, 6, 9))(strg)
        assert_equal(test, ['1', '3  4', '5  6'])

class TestNameValidator:

    def test_case_sensitivity(self):
        if False:
            i = 10
            return i + 15
        'Test case sensitivity'
        names = ['A', 'a', 'b', 'c']
        test = NameValidator().validate(names)
        assert_equal(test, ['A', 'a', 'b', 'c'])
        test = NameValidator(case_sensitive=False).validate(names)
        assert_equal(test, ['A', 'A_1', 'B', 'C'])
        test = NameValidator(case_sensitive='upper').validate(names)
        assert_equal(test, ['A', 'A_1', 'B', 'C'])
        test = NameValidator(case_sensitive='lower').validate(names)
        assert_equal(test, ['a', 'a_1', 'b', 'c'])
        assert_raises(ValueError, NameValidator, case_sensitive='foobar')

    def test_excludelist(self):
        if False:
            while True:
                i = 10
        'Test excludelist'
        names = ['dates', 'data', 'Other Data', 'mask']
        validator = NameValidator(excludelist=['dates', 'data', 'mask'])
        test = validator.validate(names)
        assert_equal(test, ['dates_', 'data_', 'Other_Data', 'mask_'])

    def test_missing_names(self):
        if False:
            print('Hello World!')
        'Test validate missing names'
        namelist = ('a', 'b', 'c')
        validator = NameValidator()
        assert_equal(validator(namelist), ['a', 'b', 'c'])
        namelist = ('', 'b', 'c')
        assert_equal(validator(namelist), ['f0', 'b', 'c'])
        namelist = ('a', 'b', '')
        assert_equal(validator(namelist), ['a', 'b', 'f0'])
        namelist = ('', 'f0', '')
        assert_equal(validator(namelist), ['f1', 'f0', 'f2'])

    def test_validate_nb_names(self):
        if False:
            return 10
        'Test validate nb names'
        namelist = ('a', 'b', 'c')
        validator = NameValidator()
        assert_equal(validator(namelist, nbfields=1), ('a',))
        assert_equal(validator(namelist, nbfields=5, defaultfmt='g%i'), ['a', 'b', 'c', 'g0', 'g1'])

    def test_validate_wo_names(self):
        if False:
            return 10
        'Test validate no names'
        namelist = None
        validator = NameValidator()
        assert_(validator(namelist) is None)
        assert_equal(validator(namelist, nbfields=3), ['f0', 'f1', 'f2'])

def _bytes_to_date(s):
    if False:
        i = 10
        return i + 15
    return date(*time.strptime(s, '%Y-%m-%d')[:3])

class TestStringConverter:
    """Test StringConverter"""

    def test_creation(self):
        if False:
            while True:
                i = 10
        'Test creation of a StringConverter'
        converter = StringConverter(int, -99999)
        assert_equal(converter._status, 1)
        assert_equal(converter.default, -99999)

    def test_upgrade(self):
        if False:
            i = 10
            return i + 15
        'Tests the upgrade method.'
        converter = StringConverter()
        assert_equal(converter._status, 0)
        assert_equal(converter.upgrade('0'), 0)
        assert_equal(converter._status, 1)
        import numpy._core.numeric as nx
        status_offset = int(nx.dtype(nx.int_).itemsize < nx.dtype(nx.int64).itemsize)
        assert_equal(converter.upgrade('17179869184'), 17179869184)
        assert_equal(converter._status, 1 + status_offset)
        assert_allclose(converter.upgrade('0.'), 0.0)
        assert_equal(converter._status, 2 + status_offset)
        assert_equal(converter.upgrade('0j'), complex('0j'))
        assert_equal(converter._status, 3 + status_offset)
        for s in ['a', b'a']:
            res = converter.upgrade(s)
            assert_(type(res) is str)
            assert_equal(res, 'a')
            assert_equal(converter._status, 8 + status_offset)

    def test_missing(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the use of missing values.'
        converter = StringConverter(missing_values=('missing', 'missed'))
        converter.upgrade('0')
        assert_equal(converter('0'), 0)
        assert_equal(converter(''), converter.default)
        assert_equal(converter('missing'), converter.default)
        assert_equal(converter('missed'), converter.default)
        try:
            converter('miss')
        except ValueError:
            pass

    def test_upgrademapper(self):
        if False:
            print('Hello World!')
        'Tests updatemapper'
        dateparser = _bytes_to_date
        _original_mapper = StringConverter._mapper[:]
        try:
            StringConverter.upgrade_mapper(dateparser, date(2000, 1, 1))
            convert = StringConverter(dateparser, date(2000, 1, 1))
            test = convert('2001-01-01')
            assert_equal(test, date(2001, 1, 1))
            test = convert('2009-01-01')
            assert_equal(test, date(2009, 1, 1))
            test = convert('')
            assert_equal(test, date(2000, 1, 1))
        finally:
            StringConverter._mapper = _original_mapper

    def test_string_to_object(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure that string-to-object functions are properly recognized'
        old_mapper = StringConverter._mapper[:]
        conv = StringConverter(_bytes_to_date)
        assert_equal(conv._mapper, old_mapper)
        assert_(hasattr(conv, 'default'))

    def test_keep_default(self):
        if False:
            for i in range(10):
                print('nop')
        "Make sure we don't lose an explicit default"
        converter = StringConverter(None, missing_values='', default=-999)
        converter.upgrade('3.14159265')
        assert_equal(converter.default, -999)
        assert_equal(converter.type, np.dtype(float))
        converter = StringConverter(None, missing_values='', default=0)
        converter.upgrade('3.14159265')
        assert_equal(converter.default, 0)
        assert_equal(converter.type, np.dtype(float))

    def test_keep_default_zero(self):
        if False:
            while True:
                i = 10
        "Check that we don't lose a default of 0"
        converter = StringConverter(int, default=0, missing_values='N/A')
        assert_equal(converter.default, 0)

    def test_keep_missing_values(self):
        if False:
            print('Hello World!')
        "Check that we're not losing missing values"
        converter = StringConverter(int, default=0, missing_values='N/A')
        assert_equal(converter.missing_values, {'', 'N/A'})

    def test_int64_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that int64 integer types can be specified'
        converter = StringConverter(np.int64, default=0)
        val = '-9223372036854775807'
        assert_(converter(val) == -9223372036854775807)
        val = '9223372036854775807'
        assert_(converter(val) == 9223372036854775807)

    def test_uint64_dtype(self):
        if False:
            return 10
        'Check that uint64 integer types can be specified'
        converter = StringConverter(np.uint64, default=0)
        val = '9223372043271415339'
        assert_(converter(val) == 9223372043271415339)

class TestMiscFunctions:

    def test_has_nested_dtype(self):
        if False:
            while True:
                i = 10
        'Test has_nested_dtype'
        ndtype = np.dtype(float)
        assert_equal(has_nested_fields(ndtype), False)
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        assert_equal(has_nested_fields(ndtype), False)
        ndtype = np.dtype([('A', int), ('B', [('BA', float), ('BB', '|S1')])])
        assert_equal(has_nested_fields(ndtype), True)

    def test_easy_dtype(self):
        if False:
            i = 10
            return i + 15
        'Test ndtype on dtypes'
        ndtype = float
        assert_equal(easy_dtype(ndtype), np.dtype(float))
        ndtype = 'i4, f8'
        assert_equal(easy_dtype(ndtype), np.dtype([('f0', 'i4'), ('f1', 'f8')]))
        assert_equal(easy_dtype(ndtype, defaultfmt='field_%03i'), np.dtype([('field_000', 'i4'), ('field_001', 'f8')]))
        ndtype = 'i4, f8'
        assert_equal(easy_dtype(ndtype, names='a, b'), np.dtype([('a', 'i4'), ('b', 'f8')]))
        ndtype = 'i4, f8'
        assert_equal(easy_dtype(ndtype, names='a, b, c'), np.dtype([('a', 'i4'), ('b', 'f8')]))
        ndtype = 'i4, f8'
        assert_equal(easy_dtype(ndtype, names=', b'), np.dtype([('f0', 'i4'), ('b', 'f8')]))
        assert_equal(easy_dtype(ndtype, names='a', defaultfmt='f%02i'), np.dtype([('a', 'i4'), ('f00', 'f8')]))
        ndtype = [('A', int), ('B', float)]
        assert_equal(easy_dtype(ndtype), np.dtype([('A', int), ('B', float)]))
        assert_equal(easy_dtype(ndtype, names='a,b'), np.dtype([('a', int), ('b', float)]))
        assert_equal(easy_dtype(ndtype, names='a'), np.dtype([('a', int), ('f0', float)]))
        assert_equal(easy_dtype(ndtype, names='a,b,c'), np.dtype([('a', int), ('b', float)]))
        ndtype = (int, float, float)
        assert_equal(easy_dtype(ndtype), np.dtype([('f0', int), ('f1', float), ('f2', float)]))
        ndtype = (int, float, float)
        assert_equal(easy_dtype(ndtype, names='a, b, c'), np.dtype([('a', int), ('b', float), ('c', float)]))
        ndtype = np.dtype(float)
        assert_equal(easy_dtype(ndtype, names='a, b, c'), np.dtype([(_, float) for _ in ('a', 'b', 'c')]))
        ndtype = np.dtype(float)
        assert_equal(easy_dtype(ndtype, names=['', '', ''], defaultfmt='f%02i'), np.dtype([(_, float) for _ in ('f00', 'f01', 'f02')]))

    def test_flatten_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        'Testing flatten_dtype'
        dt = np.dtype([('a', 'f8'), ('b', 'f8')])
        dt_flat = flatten_dtype(dt)
        assert_equal(dt_flat, [float, float])
        dt = np.dtype([('a', [('aa', '|S1'), ('ab', '|S2')]), ('b', int)])
        dt_flat = flatten_dtype(dt)
        assert_equal(dt_flat, [np.dtype('|S1'), np.dtype('|S2'), int])
        dt = np.dtype([('a', (float, 2)), ('b', (int, 3))])
        dt_flat = flatten_dtype(dt)
        assert_equal(dt_flat, [float, int])
        dt_flat = flatten_dtype(dt, True)
        assert_equal(dt_flat, [float] * 2 + [int] * 3)
        dt = np.dtype([(('a', 'A'), 'f8'), (('b', 'B'), 'f8')])
        dt_flat = flatten_dtype(dt)
        assert_equal(dt_flat, [float, float])