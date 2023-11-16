import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
import dateutil
import numpy as np
import pytest
import pytz
import pandas._libs.json as ujson
from pandas.compat import IS64
from pandas import DataFrame, DatetimeIndex, Index, NaT, PeriodIndex, Series, Timedelta, Timestamp, date_range
import pandas._testing as tm

def _clean_dict(d):
    if False:
        i = 10
        return i + 15
    '\n    Sanitize dictionary for JSON by converting all keys to strings.\n\n    Parameters\n    ----------\n    d : dict\n        The dictionary to convert.\n\n    Returns\n    -------\n    cleaned_dict : dict\n    '
    return {str(k): v for (k, v) in d.items()}

@pytest.fixture(params=[None, 'split', 'records', 'values', 'index'])
def orient(request):
    if False:
        for i in range(10):
            print('nop')
    return request.param

class TestUltraJSONTests:

    @pytest.mark.skipif(not IS64, reason='not compliant on 32-bit, xref #15865')
    def test_encode_decimal(self):
        if False:
            for i in range(10):
                print('nop')
        sut = decimal.Decimal('1337.1337')
        encoded = ujson.ujson_dumps(sut, double_precision=15)
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1337.1337
        sut = decimal.Decimal('0.95')
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == '1.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0
        sut = decimal.Decimal('0.94')
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == '0.9'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 0.9
        sut = decimal.Decimal('1.95')
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == '2.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 2.0
        sut = decimal.Decimal('-1.95')
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == '-2.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == -2.0
        sut = decimal.Decimal('0.995')
        encoded = ujson.ujson_dumps(sut, double_precision=2)
        assert encoded == '1.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0
        sut = decimal.Decimal('0.9995')
        encoded = ujson.ujson_dumps(sut, double_precision=3)
        assert encoded == '1.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0
        sut = decimal.Decimal('0.99999999999999944')
        encoded = ujson.ujson_dumps(sut, double_precision=15)
        assert encoded == '1.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0

    @pytest.mark.parametrize('ensure_ascii', [True, False])
    def test_encode_string_conversion(self, ensure_ascii):
        if False:
            return 10
        string_input = 'A string \\ / \x08 \x0c \n \r \t </script> &'
        not_html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t <\\/script> &"'
        html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t \\u003c\\/script\\u003e \\u0026"'

        def helper(expected_output, **encode_kwargs):
            if False:
                i = 10
                return i + 15
            output = ujson.ujson_dumps(string_input, ensure_ascii=ensure_ascii, **encode_kwargs)
            assert output == expected_output
            assert string_input == json.loads(output)
            assert string_input == ujson.ujson_loads(output)
        helper(not_html_encoded)
        helper(not_html_encoded, encode_html_chars=False)
        helper(html_encoded, encode_html_chars=True)

    @pytest.mark.parametrize('long_number', [-4342969734183514, -12345678901234.568, -528656961.4399388])
    def test_double_long_numbers(self, long_number):
        if False:
            print('Hello World!')
        sut = {'a': long_number}
        encoded = ujson.ujson_dumps(sut, double_precision=15)
        decoded = ujson.ujson_loads(encoded)
        assert sut == decoded

    def test_encode_non_c_locale(self):
        if False:
            print('Hello World!')
        lc_category = locale.LC_NUMERIC
        for new_locale in ('it_IT.UTF-8', 'Italian_Italy'):
            if tm.can_set_locale(new_locale, lc_category):
                with tm.set_locale(new_locale, lc_category):
                    assert ujson.ujson_loads(ujson.ujson_dumps(4.78e+60)) == 4.78e+60
                    assert ujson.ujson_loads('4.78', precise_float=True) == 4.78
                break

    def test_decimal_decode_test_precise(self):
        if False:
            print('Hello World!')
        sut = {'a': 4.56}
        encoded = ujson.ujson_dumps(sut)
        decoded = ujson.ujson_loads(encoded, precise_float=True)
        assert sut == decoded

    def test_encode_double_tiny_exponential(self):
        if False:
            print('Hello World!')
        num = 1e-40
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = 1e-100
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-45
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-145
        assert np.allclose(num, ujson.ujson_loads(ujson.ujson_dumps(num)))

    @pytest.mark.parametrize('unicode_key', ['key1', 'بن'])
    def test_encode_dict_with_unicode_keys(self, unicode_key):
        if False:
            for i in range(10):
                print('nop')
        unicode_dict = {unicode_key: 'value1'}
        assert unicode_dict == ujson.ujson_loads(ujson.ujson_dumps(unicode_dict))

    @pytest.mark.parametrize('double_input', [math.pi, -math.pi])
    def test_encode_double_conversion(self, double_input):
        if False:
            while True:
                i = 10
        output = ujson.ujson_dumps(double_input)
        assert round(double_input, 5) == round(json.loads(output), 5)
        assert round(double_input, 5) == round(ujson.ujson_loads(output), 5)

    def test_encode_with_decimal(self):
        if False:
            for i in range(10):
                print('nop')
        decimal_input = 1.0
        output = ujson.ujson_dumps(decimal_input)
        assert output == '1.0'

    def test_encode_array_of_nested_arrays(self):
        if False:
            print('Hello World!')
        nested_input = [[[[]]]] * 20
        output = ujson.ujson_dumps(nested_input)
        assert nested_input == json.loads(output)
        assert nested_input == ujson.ujson_loads(output)

    def test_encode_array_of_doubles(self):
        if False:
            print('Hello World!')
        doubles_input = [31337.31337, 31337.31337, 31337.31337, 31337.31337] * 10
        output = ujson.ujson_dumps(doubles_input)
        assert doubles_input == json.loads(output)
        assert doubles_input == ujson.ujson_loads(output)

    def test_double_precision(self):
        if False:
            print('Hello World!')
        double_input = 30.012345678901234
        output = ujson.ujson_dumps(double_input, double_precision=15)
        assert double_input == json.loads(output)
        assert double_input == ujson.ujson_loads(output)
        for double_precision in (3, 9):
            output = ujson.ujson_dumps(double_input, double_precision=double_precision)
            rounded_input = round(double_input, double_precision)
            assert rounded_input == json.loads(output)
            assert rounded_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('invalid_val', [20, -1, '9', None])
    def test_invalid_double_precision(self, invalid_val):
        if False:
            print('Hello World!')
        double_input = 30.123456789012344
        expected_exception = ValueError if isinstance(invalid_val, int) else TypeError
        msg = "Invalid value '.*' for option 'double_precision', max is '15'|an integer is required \\(got type |object cannot be interpreted as an integer"
        with pytest.raises(expected_exception, match=msg):
            ujson.ujson_dumps(double_input, double_precision=invalid_val)

    def test_encode_string_conversion2(self):
        if False:
            while True:
                i = 10
        string_input = 'A string \\ / \x08 \x0c \n \r \t'
        output = ujson.ujson_dumps(string_input)
        assert string_input == json.loads(output)
        assert string_input == ujson.ujson_loads(output)
        assert output == '"A string \\\\ \\/ \\b \\f \\n \\r \\t"'

    @pytest.mark.parametrize('unicode_input', ['Räksmörgås اسامة بن محمد بن عوض بن لادن', 'æ\x97¥Ñ\x88'])
    def test_encode_unicode_conversion(self, unicode_input):
        if False:
            i = 10
            return i + 15
        enc = ujson.ujson_dumps(unicode_input)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(unicode_input)
        assert dec == json.loads(enc)

    def test_encode_control_escaping(self):
        if False:
            print('Hello World!')
        escaped_input = '\x19'
        enc = ujson.ujson_dumps(escaped_input)
        dec = ujson.ujson_loads(enc)
        assert escaped_input == dec
        assert enc == json.dumps(escaped_input)

    def test_encode_unicode_surrogate_pair(self):
        if False:
            i = 10
            return i + 15
        surrogate_input = 'ð\x90\x8d\x86'
        enc = ujson.ujson_dumps(surrogate_input)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(surrogate_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8(self):
        if False:
            for i in range(10):
                print('nop')
        four_bytes_input = 'ð\x91\x80°TRAILINGNORMAL'
        enc = ujson.ujson_dumps(four_bytes_input)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8highest(self):
        if False:
            return 10
        four_bytes_input = 'ó¿¿¿TRAILINGNORMAL'
        enc = ujson.ujson_dumps(four_bytes_input)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_error(self):
        if False:
            while True:
                i = 10
        string = "'\udac0'"
        msg = "'utf-8' codec can't encode character '\\\\udac0' in position 1: surrogates not allowed"
        with pytest.raises(UnicodeEncodeError, match=msg):
            ujson.ujson_dumps([string])

    def test_encode_array_in_array(self):
        if False:
            for i in range(10):
                print('nop')
        arr_in_arr_input = [[[[]]]]
        output = ujson.ujson_dumps(arr_in_arr_input)
        assert arr_in_arr_input == json.loads(output)
        assert output == json.dumps(arr_in_arr_input)
        assert arr_in_arr_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('num_input', [31337, -31337, -9223372036854775808])
    def test_encode_num_conversion(self, num_input):
        if False:
            for i in range(10):
                print('nop')
        output = ujson.ujson_dumps(num_input)
        assert num_input == json.loads(output)
        assert output == json.dumps(num_input)
        assert num_input == ujson.ujson_loads(output)

    def test_encode_list_conversion(self):
        if False:
            for i in range(10):
                print('nop')
        list_input = [1, 2, 3, 4]
        output = ujson.ujson_dumps(list_input)
        assert list_input == json.loads(output)
        assert list_input == ujson.ujson_loads(output)

    def test_encode_dict_conversion(self):
        if False:
            return 10
        dict_input = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4}
        output = ujson.ujson_dumps(dict_input)
        assert dict_input == json.loads(output)
        assert dict_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('builtin_value', [None, True, False])
    def test_encode_builtin_values_conversion(self, builtin_value):
        if False:
            i = 10
            return i + 15
        output = ujson.ujson_dumps(builtin_value)
        assert builtin_value == json.loads(output)
        assert output == json.dumps(builtin_value)
        assert builtin_value == ujson.ujson_loads(output)

    def test_encode_datetime_conversion(self):
        if False:
            return 10
        datetime_input = datetime.datetime.fromtimestamp(time.time())
        output = ujson.ujson_dumps(datetime_input, date_unit='s')
        expected = calendar.timegm(datetime_input.utctimetuple())
        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    def test_encode_date_conversion(self):
        if False:
            return 10
        date_input = datetime.date.fromtimestamp(time.time())
        output = ujson.ujson_dumps(date_input, date_unit='s')
        tup = (date_input.year, date_input.month, date_input.day, 0, 0, 0)
        expected = calendar.timegm(tup)
        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    @pytest.mark.parametrize('test', [datetime.time(), datetime.time(1, 2, 3), datetime.time(10, 12, 15, 343243)])
    def test_encode_time_conversion_basic(self, test):
        if False:
            return 10
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_pytz(self):
        if False:
            while True:
                i = 10
        test = datetime.time(10, 12, 15, 343243, pytz.utc)
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_dateutil(self):
        if False:
            for i in range(10):
                print('nop')
        test = datetime.time(10, 12, 15, 343243, dateutil.tz.tzutc())
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    @pytest.mark.parametrize('decoded_input', [NaT, np.datetime64('NaT'), np.nan, np.inf, -np.inf])
    def test_encode_as_null(self, decoded_input):
        if False:
            print('Hello World!')
        assert ujson.ujson_dumps(decoded_input) == 'null', 'Expected null'

    def test_datetime_units(self):
        if False:
            i = 10
            return i + 15
        val = datetime.datetime(2013, 8, 17, 21, 17, 12, 215504)
        stamp = Timestamp(val).as_unit('ns')
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='s'))
        assert roundtrip == stamp._value // 10 ** 9
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='ms'))
        assert roundtrip == stamp._value // 10 ** 6
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='us'))
        assert roundtrip == stamp._value // 10 ** 3
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='ns'))
        assert roundtrip == stamp._value
        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_dumps(val, date_unit='foo')

    def test_encode_to_utf8(self):
        if False:
            print('Hello World!')
        unencoded = 'æ\x97¥Ñ\x88'
        enc = ujson.ujson_dumps(unencoded, ensure_ascii=False)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(unencoded, ensure_ascii=False)
        assert dec == json.loads(enc)

    def test_decode_from_unicode(self):
        if False:
            return 10
        unicode_input = '{"obj": 31337}'
        dec1 = ujson.ujson_loads(unicode_input)
        dec2 = ujson.ujson_loads(str(unicode_input))
        assert dec1 == dec2

    def test_encode_recursion_max(self):
        if False:
            i = 10
            return i + 15

        class O2:
            member = 0

        class O1:
            member = 0
        decoded_input = O1()
        decoded_input.member = O2()
        decoded_input.member.member = decoded_input
        with pytest.raises(OverflowError, match='Maximum recursion level reached'):
            ujson.ujson_dumps(decoded_input)

    def test_decode_jibberish(self):
        if False:
            return 10
        jibberish = 'fdsa sda v9sa fdsa'
        msg = "Unexpected character found when decoding 'false'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(jibberish)

    @pytest.mark.parametrize('broken_json', ['[', '{', ']', '}'])
    def test_decode_broken_json(self, broken_json):
        if False:
            print('Hello World!')
        msg = 'Expected object or value'
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize('too_big_char', ['[', '{'])
    def test_decode_depth_too_big(self, too_big_char):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='Reached object decoding depth limit'):
            ujson.ujson_loads(too_big_char * (1024 * 1024))

    @pytest.mark.parametrize('bad_string', ['"TESTING', '"TESTING\\"', 'tru', 'fa', 'n'])
    def test_decode_bad_string(self, bad_string):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Unexpected character found when decoding|Unmatched \'\'"\' when when decoding \'string\''
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(bad_string)

    @pytest.mark.parametrize('broken_json, err_msg', [('{{1337:""}}', "Key name of object must be 'string' when decoding 'object'"), ('{{"key":"}', 'Unmatched \'\'"\' when when decoding \'string\''), ('[[[true', 'Unexpected character found when decoding array value (2)')])
    def test_decode_broken_json_leak(self, broken_json, err_msg):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(1000):
            with pytest.raises(ValueError, match=re.escape(err_msg)):
                ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize('invalid_dict', ['{{{{31337}}}}', '{{{{"key":}}}}', '{{{{"key"}}}}'])
    def test_decode_invalid_dict(self, invalid_dict):
        if False:
            return 10
        msg = "Key name of object must be 'string' when decoding 'object'|No ':' found when decoding object value|Expected object or value"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_dict)

    @pytest.mark.parametrize('numeric_int_as_str', ['31337', '-31337'])
    def test_decode_numeric_int(self, numeric_int_as_str):
        if False:
            while True:
                i = 10
        assert int(numeric_int_as_str) == ujson.ujson_loads(numeric_int_as_str)

    def test_encode_null_character(self):
        if False:
            for i in range(10):
                print('nop')
        wrapped_input = '31337 \x00 1337'
        output = ujson.ujson_dumps(wrapped_input)
        assert wrapped_input == json.loads(output)
        assert output == json.dumps(wrapped_input)
        assert wrapped_input == ujson.ujson_loads(output)
        alone_input = '\x00'
        output = ujson.ujson_dumps(alone_input)
        assert alone_input == json.loads(output)
        assert output == json.dumps(alone_input)
        assert alone_input == ujson.ujson_loads(output)
        assert '"  \\u0000\\r\\n "' == ujson.ujson_dumps('  \x00\r\n ')

    def test_decode_null_character(self):
        if False:
            while True:
                i = 10
        wrapped_input = '"31337 \\u0000 31337"'
        assert ujson.ujson_loads(wrapped_input) == json.loads(wrapped_input)

    def test_encode_list_long_conversion(self):
        if False:
            print('Hello World!')
        long_input = [9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807]
        output = ujson.ujson_dumps(long_input)
        assert long_input == json.loads(output)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('long_input', [9223372036854775807, 18446744073709551615])
    def test_encode_long_conversion(self, long_input):
        if False:
            return 10
        output = ujson.ujson_dumps(long_input)
        assert long_input == json.loads(output)
        assert output == json.dumps(long_input)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('bigNum', [2 ** 64, -2 ** 63 - 1])
    def test_dumps_ints_larger_than_maxsize(self, bigNum):
        if False:
            print('Hello World!')
        encoding = ujson.ujson_dumps(bigNum)
        assert str(bigNum) == encoding
        with pytest.raises(ValueError, match='Value is too big|Value is too small'):
            assert ujson.ujson_loads(encoding) == bigNum

    @pytest.mark.parametrize('int_exp', ['1337E40', '1.337E40', '1337E+9', '1.337e+40', '1.337E-4'])
    def test_decode_numeric_int_exp(self, int_exp):
        if False:
            print('Hello World!')
        assert ujson.ujson_loads(int_exp) == json.loads(int_exp)

    def test_loads_non_str_bytes_raises(self):
        if False:
            i = 10
            return i + 15
        msg = "Expected 'str' or 'bytes'"
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_loads(None)

    @pytest.mark.parametrize('val', [3590016419, 2 ** 31, 2 ** 32, 2 ** 32 - 1])
    def test_decode_number_with_32bit_sign_bit(self, val):
        if False:
            while True:
                i = 10
        doc = f'{{"id": {val}}}'
        assert ujson.ujson_loads(doc)['id'] == val

    def test_encode_big_escape(self):
        if False:
            while True:
                i = 10
        for _ in range(10):
            base = 'å'.encode()
            escape_input = base * 1024 * 1024 * 2
            ujson.ujson_dumps(escape_input)

    def test_decode_big_escape(self):
        if False:
            while True:
                i = 10
        for _ in range(10):
            base = 'å'.encode()
            quote = b'"'
            escape_input = quote + base * 1024 * 1024 * 2 + quote
            ujson.ujson_loads(escape_input)

    def test_to_dict(self):
        if False:
            while True:
                i = 10
        d = {'key': 31337}

        class DictTest:

            def toDict(self):
                if False:
                    print('Hello World!')
                return d
        o = DictTest()
        output = ujson.ujson_dumps(o)
        dec = ujson.ujson_loads(output)
        assert dec == d

    def test_default_handler(self):
        if False:
            for i in range(10):
                print('nop')

        class _TestObject:

            def __init__(self, val) -> None:
                if False:
                    return 10
                self.val = val

            @property
            def recursive_attr(self):
                if False:
                    print('Hello World!')
                return _TestObject('recursive_attr')

            def __str__(self) -> str:
                if False:
                    return 10
                return str(self.val)
        msg = 'Maximum recursion level reached'
        with pytest.raises(OverflowError, match=msg):
            ujson.ujson_dumps(_TestObject('foo'))
        assert '"foo"' == ujson.ujson_dumps(_TestObject('foo'), default_handler=str)

        def my_handler(_):
            if False:
                while True:
                    i = 10
            return 'foobar'
        assert '"foobar"' == ujson.ujson_dumps(_TestObject('foo'), default_handler=my_handler)

        def my_handler_raises(_):
            if False:
                for i in range(10):
                    print('nop')
            raise TypeError('I raise for anything')
        with pytest.raises(TypeError, match='I raise for anything'):
            ujson.ujson_dumps(_TestObject('foo'), default_handler=my_handler_raises)

        def my_int_handler(_):
            if False:
                i = 10
                return i + 15
            return 42
        assert ujson.ujson_loads(ujson.ujson_dumps(_TestObject('foo'), default_handler=my_int_handler)) == 42

        def my_obj_handler(_):
            if False:
                for i in range(10):
                    print('nop')
            return datetime.datetime(2013, 2, 3)
        assert ujson.ujson_loads(ujson.ujson_dumps(datetime.datetime(2013, 2, 3))) == ujson.ujson_loads(ujson.ujson_dumps(_TestObject('foo'), default_handler=my_obj_handler))
        obj_list = [_TestObject('foo'), _TestObject('bar')]
        assert json.loads(json.dumps(obj_list, default=str)) == ujson.ujson_loads(ujson.ujson_dumps(obj_list, default_handler=str))

    def test_encode_object(self):
        if False:
            for i in range(10):
                print('nop')

        class _TestObject:

            def __init__(self, a, b, _c, d) -> None:
                if False:
                    return 10
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self):
                if False:
                    return 10
                return 5
        test_object = _TestObject(a=1, b=2, _c=3, d=4)
        assert ujson.ujson_loads(ujson.ujson_dumps(test_object)) == {'a': 1, 'b': 2, 'd': 4}

    def test_ujson__name__(self):
        if False:
            print('Hello World!')
        assert ujson.__name__ == 'pandas._libs.json'

class TestNumpyJSONTests:

    @pytest.mark.parametrize('bool_input', [True, False])
    def test_bool(self, bool_input):
        if False:
            print('Hello World!')
        b = bool(bool_input)
        assert ujson.ujson_loads(ujson.ujson_dumps(b)) == b

    def test_bool_array(self):
        if False:
            while True:
                i = 10
        bool_array = np.array([True, False, True, True, False, True, False, False], dtype=bool)
        output = np.array(ujson.ujson_loads(ujson.ujson_dumps(bool_array)), dtype=bool)
        tm.assert_numpy_array_equal(bool_array, output)

    def test_int(self, any_int_numpy_dtype):
        if False:
            i = 10
            return i + 15
        klass = np.dtype(any_int_numpy_dtype).type
        num = klass(1)
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_int_array(self, any_int_numpy_dtype):
        if False:
            i = 10
            return i + 15
        arr = np.arange(100, dtype=int)
        arr_input = arr.astype(any_int_numpy_dtype)
        arr_output = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr_input)), dtype=any_int_numpy_dtype)
        tm.assert_numpy_array_equal(arr_input, arr_output)

    def test_int_max(self, any_int_numpy_dtype):
        if False:
            return 10
        if any_int_numpy_dtype in ('int64', 'uint64') and (not IS64):
            pytest.skip('Cannot test 64-bit integer on 32-bit platform')
        klass = np.dtype(any_int_numpy_dtype).type
        if any_int_numpy_dtype == 'uint64':
            num = np.iinfo('int64').max
        else:
            num = np.iinfo(any_int_numpy_dtype).max
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float(self, float_numpy_dtype):
        if False:
            i = 10
            return i + 15
        klass = np.dtype(float_numpy_dtype).type
        num = klass(256.2013)
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float_array(self, float_numpy_dtype):
        if False:
            print('Hello World!')
        arr = np.arange(12.5, 185.72, 1.7322, dtype=float)
        float_input = arr.astype(float_numpy_dtype)
        float_output = np.array(ujson.ujson_loads(ujson.ujson_dumps(float_input, double_precision=15)), dtype=float_numpy_dtype)
        tm.assert_almost_equal(float_input, float_output)

    def test_float_max(self, float_numpy_dtype):
        if False:
            print('Hello World!')
        klass = np.dtype(float_numpy_dtype).type
        num = klass(np.finfo(float_numpy_dtype).max / 10)
        tm.assert_almost_equal(klass(ujson.ujson_loads(ujson.ujson_dumps(num, double_precision=15))), num)

    def test_array_basic(self):
        if False:
            for i in range(10):
                print('nop')
        arr = np.arange(96)
        arr = arr.reshape((2, 2, 2, 2, 3, 2))
        tm.assert_numpy_array_equal(np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr)

    @pytest.mark.parametrize('shape', [(10, 10), (5, 5, 4), (100, 1)])
    def test_array_reshaped(self, shape):
        if False:
            for i in range(10):
                print('nop')
        arr = np.arange(100)
        arr = arr.reshape(shape)
        tm.assert_numpy_array_equal(np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr)

    def test_array_list(self):
        if False:
            for i in range(10):
                print('nop')
        arr_list = ['a', [], {}, {}, [], 42, 97.8, ['a', 'b'], {'key': 'val'}]
        arr = np.array(arr_list, dtype=object)
        result = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=object)
        tm.assert_numpy_array_equal(result, arr)

    def test_array_float(self):
        if False:
            print('Hello World!')
        dtype = np.float32
        arr = np.arange(100.202, 200.202, 1, dtype=dtype)
        arr = arr.reshape((5, 5, 4))
        arr_out = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=dtype)
        tm.assert_almost_equal(arr, arr_out)

    def test_0d_array(self):
        if False:
            for i in range(10):
                print('nop')
        msg = re.escape('array(1) (numpy-scalar) is not JSON serializable at the moment')
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.array(1))

    def test_array_long_double(self):
        if False:
            for i in range(10):
                print('nop')
        msg = re.compile('1234.5.* \\(numpy-scalar\\) is not JSON serializable at the moment')
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.longdouble(1234.5))

class TestPandasJSONTests:

    def test_dataframe(self, orient):
        if False:
            return 10
        dtype = np.int64
        df = DataFrame([[1, 2, 3], [4, 5, 6]], index=['a', 'b'], columns=['x', 'y', 'z'], dtype=dtype)
        encode_kwargs = {} if orient is None else {'orient': orient}
        assert (df.dtypes == dtype).all()
        output = ujson.ujson_loads(ujson.ujson_dumps(df, **encode_kwargs))
        assert (df.dtypes == dtype).all()
        if orient == 'split':
            dec = _clean_dict(output)
            output = DataFrame(**dec)
        else:
            output = DataFrame(output)
        if orient == 'values':
            df.columns = [0, 1, 2]
            df.index = [0, 1]
        elif orient == 'records':
            df.index = [0, 1]
        elif orient == 'index':
            df = df.transpose()
        assert (df.dtypes == dtype).all()
        tm.assert_frame_equal(output, df)

    def test_dataframe_nested(self, orient):
        if False:
            i = 10
            return i + 15
        df = DataFrame([[1, 2, 3], [4, 5, 6]], index=['a', 'b'], columns=['x', 'y', 'z'])
        nested = {'df1': df, 'df2': df.copy()}
        kwargs = {} if orient is None else {'orient': orient}
        exp = {'df1': ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)), 'df2': ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs))}
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_series(self, orient):
        if False:
            while True:
                i = 10
        dtype = np.int64
        s = Series([10, 20, 30, 40, 50, 60], name='series', index=[6, 7, 8, 9, 10, 15], dtype=dtype).sort_values()
        assert s.dtype == dtype
        encode_kwargs = {} if orient is None else {'orient': orient}
        output = ujson.ujson_loads(ujson.ujson_dumps(s, **encode_kwargs))
        assert s.dtype == dtype
        if orient == 'split':
            dec = _clean_dict(output)
            output = Series(**dec)
        else:
            output = Series(output)
        if orient in (None, 'index'):
            s.name = None
            output = output.sort_values()
            s.index = ['6', '7', '8', '9', '10', '15']
        elif orient in ('records', 'values'):
            s.name = None
            s.index = [0, 1, 2, 3, 4, 5]
        assert s.dtype == dtype
        tm.assert_series_equal(output, s)

    def test_series_nested(self, orient):
        if False:
            return 10
        s = Series([10, 20, 30, 40, 50, 60], name='series', index=[6, 7, 8, 9, 10, 15]).sort_values()
        nested = {'s1': s, 's2': s.copy()}
        kwargs = {} if orient is None else {'orient': orient}
        exp = {'s1': ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)), 's2': ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs))}
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_index(self):
        if False:
            return 10
        i = Index([23, 45, 18, 98, 43, 11], name='index')
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i)), name='index')
        tm.assert_index_equal(i, output)
        dec = _clean_dict(ujson.ujson_loads(ujson.ujson_dumps(i, orient='split')))
        output = Index(**dec)
        tm.assert_index_equal(i, output)
        assert i.name == output.name
        tm.assert_index_equal(i, output)
        assert i.name == output.name
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i, orient='values')), name='index')
        tm.assert_index_equal(i, output)
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i, orient='records')), name='index')
        tm.assert_index_equal(i, output)
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i, orient='index')), name='index')
        tm.assert_index_equal(i, output)

    def test_datetime_index(self):
        if False:
            print('Hello World!')
        date_unit = 'ns'
        rng = DatetimeIndex(list(date_range('1/1/2000', periods=20)), freq=None)
        encoded = ujson.ujson_dumps(rng, date_unit=date_unit)
        decoded = DatetimeIndex(np.array(ujson.ujson_loads(encoded)))
        tm.assert_index_equal(rng, decoded)
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        decoded = Series(ujson.ujson_loads(ujson.ujson_dumps(ts, date_unit=date_unit)))
        idx_values = decoded.index.values.astype(np.int64)
        decoded.index = DatetimeIndex(idx_values)
        tm.assert_series_equal(ts, decoded)

    @pytest.mark.parametrize('invalid_arr', ['[31337,]', '[,31337]', '[]]', '[,]'])
    def test_decode_invalid_array(self, invalid_arr):
        if False:
            return 10
        msg = 'Expected object or value|Trailing data|Unexpected character found when decoding array value'
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_arr)

    @pytest.mark.parametrize('arr', [[], [31337]])
    def test_decode_array(self, arr):
        if False:
            print('Hello World!')
        assert arr == ujson.ujson_loads(str(arr))

    @pytest.mark.parametrize('extreme_num', [9223372036854775807, -9223372036854775808])
    def test_decode_extreme_numbers(self, extreme_num):
        if False:
            while True:
                i = 10
        assert extreme_num == ujson.ujson_loads(str(extreme_num))

    @pytest.mark.parametrize('too_extreme_num', [f'{2 ** 64}', f'{-2 ** 63 - 1}'])
    def test_decode_too_extreme_numbers(self, too_extreme_num):
        if False:
            return 10
        with pytest.raises(ValueError, match='Value is too big|Value is too small'):
            ujson.ujson_loads(too_extreme_num)

    def test_decode_with_trailing_whitespaces(self):
        if False:
            while True:
                i = 10
        assert {} == ujson.ujson_loads('{}\n\t ')

    def test_decode_with_trailing_non_whitespaces(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match='Trailing data'):
            ujson.ujson_loads('{}\n\t a')

    @pytest.mark.parametrize('value', [f'{2 ** 64}', f'{-2 ** 63 - 1}'])
    def test_decode_array_with_big_int(self, value):
        if False:
            return 10
        with pytest.raises(ValueError, match='Value is too big|Value is too small'):
            ujson.ujson_loads(value)

    @pytest.mark.parametrize('float_number', [1.1234567893, 1.234567893, 1.34567893, 1.4567893, 1.567893, 1.67893, 1.7893, 1.893, 1.3])
    @pytest.mark.parametrize('sign', [-1, 1])
    def test_decode_floating_point(self, sign, float_number):
        if False:
            while True:
                i = 10
        float_number *= sign
        tm.assert_almost_equal(float_number, ujson.ujson_loads(str(float_number)), rtol=1e-15)

    def test_encode_big_set(self):
        if False:
            return 10
        s = set()
        for x in range(100000):
            s.add(x)
        ujson.ujson_dumps(s)

    def test_encode_empty_set(self):
        if False:
            i = 10
            return i + 15
        assert '[]' == ujson.ujson_dumps(set())

    def test_encode_set(self):
        if False:
            for i in range(10):
                print('nop')
        s = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        enc = ujson.ujson_dumps(s)
        dec = ujson.ujson_loads(enc)
        for v in dec:
            assert v in s

    @pytest.mark.parametrize('td', [Timedelta(days=366), Timedelta(days=-1), Timedelta(hours=13, minutes=5, seconds=5), Timedelta(hours=13, minutes=20, seconds=30), Timedelta(days=-1, nanoseconds=5), Timedelta(nanoseconds=1), Timedelta(microseconds=1, nanoseconds=1), Timedelta(milliseconds=1, microseconds=1, nanoseconds=1), Timedelta(milliseconds=999, microseconds=999, nanoseconds=999)])
    def test_encode_timedelta_iso(self, td):
        if False:
            i = 10
            return i + 15
        result = ujson.ujson_dumps(td, iso_dates=True)
        expected = f'"{td.isoformat()}"'
        assert result == expected

    def test_encode_periodindex(self):
        if False:
            i = 10
            return i + 15
        p = PeriodIndex(['2022-04-06', '2022-04-07'], freq='D')
        df = DataFrame(index=p)
        assert df.to_json() == '{}'