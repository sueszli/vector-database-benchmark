import dataclasses
import datetime
import uuid
import pytest
import orjson
try:
    import pytz
except ImportError:
    pytz = None
try:
    import numpy
except ImportError:
    numpy = None

class SubStr(str):
    pass

class TestNonStrKeyTests:

    def test_dict_keys_duplicate(self):
        if False:
            return 10
        '\n        OPT_NON_STR_KEYS serializes duplicate keys\n        '
        assert orjson.dumps({'1': True, 1: False}, option=orjson.OPT_NON_STR_KEYS) == b'{"1":true,"1":false}'

    def test_dict_keys_int(self):
        if False:
            return 10
        assert orjson.dumps({1: True, 2: False}, option=orjson.OPT_NON_STR_KEYS) == b'{"1":true,"2":false}'

    def test_dict_keys_substr(self):
        if False:
            return 10
        assert orjson.dumps({SubStr('aaa'): True}, option=orjson.OPT_NON_STR_KEYS) == b'{"aaa":true}'

    def test_dict_keys_substr_passthrough(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        OPT_PASSTHROUGH_SUBCLASS does not affect OPT_NON_STR_KEYS\n        '
        assert orjson.dumps({SubStr('aaa'): True}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_PASSTHROUGH_SUBCLASS) == b'{"aaa":true}'

    def test_dict_keys_substr_invalid(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps({SubStr('\ud800'): True}, option=orjson.OPT_NON_STR_KEYS)

    def test_dict_keys_strict(self):
        if False:
            print('Hello World!')
        '\n        OPT_NON_STR_KEYS does not respect OPT_STRICT_INTEGER\n        '
        assert orjson.dumps({9223372036854775807: True}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_STRICT_INTEGER) == b'{"9223372036854775807":true}'

    def test_dict_keys_int_range_valid_i64(self):
        if False:
            while True:
                i = 10
        '\n        OPT_NON_STR_KEYS has a i64 range for int, valid\n        '
        assert orjson.dumps({9223372036854775807: True}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_STRICT_INTEGER) == b'{"9223372036854775807":true}'
        assert orjson.dumps({-9223372036854775807: True}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_STRICT_INTEGER) == b'{"-9223372036854775807":true}'
        assert orjson.dumps({9223372036854775809: True}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_STRICT_INTEGER) == b'{"9223372036854775809":true}'

    def test_dict_keys_int_range_valid_u64(self):
        if False:
            while True:
                i = 10
        '\n        OPT_NON_STR_KEYS has a u64 range for int, valid\n        '
        assert orjson.dumps({0: True}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_STRICT_INTEGER) == b'{"0":true}'
        assert orjson.dumps({18446744073709551615: True}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_STRICT_INTEGER) == b'{"18446744073709551615":true}'

    def test_dict_keys_int_range_invalid(self):
        if False:
            return 10
        '\n        OPT_NON_STR_KEYS has a range of i64::MIN to u64::MAX\n        '
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps({-9223372036854775809: True}, option=orjson.OPT_NON_STR_KEYS)
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps({18446744073709551616: True}, option=orjson.OPT_NON_STR_KEYS)

    def test_dict_keys_float(self):
        if False:
            while True:
                i = 10
        assert orjson.dumps({1.1: True, 2.2: False}, option=orjson.OPT_NON_STR_KEYS) == b'{"1.1":true,"2.2":false}'

    def test_dict_keys_inf(self):
        if False:
            return 10
        assert orjson.dumps({float('Infinity'): True}, option=orjson.OPT_NON_STR_KEYS) == b'{"null":true}'
        assert orjson.dumps({float('-Infinity'): True}, option=orjson.OPT_NON_STR_KEYS) == b'{"null":true}'

    def test_dict_keys_nan(self):
        if False:
            i = 10
            return i + 15
        assert orjson.dumps({float('NaN'): True}, option=orjson.OPT_NON_STR_KEYS) == b'{"null":true}'

    def test_dict_keys_bool(self):
        if False:
            while True:
                i = 10
        assert orjson.dumps({True: True, False: False}, option=orjson.OPT_NON_STR_KEYS) == b'{"true":true,"false":false}'

    def test_dict_keys_datetime(self):
        if False:
            return 10
        assert orjson.dumps({datetime.datetime(2000, 1, 1, 2, 3, 4, 123): True}, option=orjson.OPT_NON_STR_KEYS) == b'{"2000-01-01T02:03:04.000123":true}'

    def test_dict_keys_datetime_opt(self):
        if False:
            i = 10
            return i + 15
        assert orjson.dumps({datetime.datetime(2000, 1, 1, 2, 3, 4, 123): True}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_OMIT_MICROSECONDS | orjson.OPT_NAIVE_UTC | orjson.OPT_UTC_Z) == b'{"2000-01-01T02:03:04Z":true}'

    def test_dict_keys_datetime_passthrough(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        OPT_PASSTHROUGH_DATETIME does not affect OPT_NON_STR_KEYS\n        '
        assert orjson.dumps({datetime.datetime(2000, 1, 1, 2, 3, 4, 123): True}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_PASSTHROUGH_DATETIME) == b'{"2000-01-01T02:03:04.000123":true}'

    def test_dict_keys_uuid(self):
        if False:
            i = 10
            return i + 15
        '\n        OPT_NON_STR_KEYS always serializes UUID as keys\n        '
        assert orjson.dumps({uuid.UUID('7202d115-7ff3-4c81-a7c1-2a1f067b1ece'): True}, option=orjson.OPT_NON_STR_KEYS) == b'{"7202d115-7ff3-4c81-a7c1-2a1f067b1ece":true}'

    def test_dict_keys_date(self):
        if False:
            return 10
        assert orjson.dumps({datetime.date(1970, 1, 1): True}, option=orjson.OPT_NON_STR_KEYS) == b'{"1970-01-01":true}'

    def test_dict_keys_time(self):
        if False:
            print('Hello World!')
        assert orjson.dumps({datetime.time(12, 15, 59, 111): True}, option=orjson.OPT_NON_STR_KEYS) == b'{"12:15:59.000111":true}'

    def test_dict_non_str_and_sort_keys(self):
        if False:
            return 10
        assert orjson.dumps({'other': 1, datetime.date(1970, 1, 5): 2, datetime.date(1970, 1, 3): 3}, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SORT_KEYS) == b'{"1970-01-03":3,"1970-01-05":2,"other":1}'

    @pytest.mark.skipif(pytz is None, reason='pytz optional')
    def test_dict_keys_time_err(self):
        if False:
            print('Hello World!')
        '\n        OPT_NON_STR_KEYS propagates errors in types\n        '
        val = datetime.time(12, 15, 59, 111, tzinfo=pytz.timezone('Asia/Shanghai'))
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps({val: True}, option=orjson.OPT_NON_STR_KEYS)

    def test_dict_keys_str(self):
        if False:
            while True:
                i = 10
        assert orjson.dumps({'1': True}, option=orjson.OPT_NON_STR_KEYS) == b'{"1":true}'

    def test_dict_keys_type(self):
        if False:
            i = 10
            return i + 15

        class Obj:
            a: str
        val = Obj()
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps({val: True}, option=orjson.OPT_NON_STR_KEYS)

    @pytest.mark.skipif(numpy is None, reason='numpy is not installed')
    def test_dict_keys_array(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError):
            {numpy.array([1, 2]): True}

    def test_dict_keys_dataclass(self):
        if False:
            while True:
                i = 10

        @dataclasses.dataclass
        class Dataclass:
            a: str
        with pytest.raises(TypeError):
            {Dataclass('a'): True}

    def test_dict_keys_dataclass_hash(self):
        if False:
            while True:
                i = 10

        @dataclasses.dataclass
        class Dataclass:
            a: str

            def __hash__(self):
                if False:
                    i = 10
                    return i + 15
                return 1
        obj = {Dataclass('a'): True}
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)

    def test_dict_keys_list(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError):
            {[]: True}

    def test_dict_keys_dict(self):
        if False:
            return 10
        with pytest.raises(TypeError):
            {{}: True}

    def test_dict_keys_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        obj = {(): True}
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)

    def test_dict_keys_unknown(self):
        if False:
            return 10
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps({frozenset(): True}, option=orjson.OPT_NON_STR_KEYS)

    def test_dict_keys_no_str_call(self):
        if False:
            return 10

        class Obj:
            a: str

            def __str__(self):
                if False:
                    i = 10
                    return i + 15
                return 'Obj'
        val = Obj()
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps({val: True}, option=orjson.OPT_NON_STR_KEYS)