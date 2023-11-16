from datetime import date, datetime
from perspective.table import Table
from pytest import mark

class TestTableInfer(object):

    def test_table_infer_int(self):
        if False:
            i = 10
            return i + 15
        data = {'a': [None, None, None, None, 1, 0, 1, 1, 1]}
        tbl = Table(data)
        assert tbl.schema() == {'a': int}

    def test_table_infer_float(self):
        if False:
            print('Hello World!')
        data = {'a': [None, None, None, None, 1.0, 2.0]}
        tbl = Table(data)
        assert tbl.schema() == {'a': float}

    def test_table_infer_bool(self):
        if False:
            i = 10
            return i + 15
        bool_data = [{'a': True, 'b': False}, {'a': True, 'b': True}]
        tbl = Table(bool_data)
        assert tbl.size() == 2
        assert tbl.schema() == {'a': bool, 'b': bool}

    def test_table_infer_bool_str(self):
        if False:
            i = 10
            return i + 15
        bool_data = [{'a': 'True', 'b': 'False'}, {'a': 'True', 'b': 'True'}]
        tbl = Table(bool_data)
        assert tbl.size() == 2
        assert tbl.schema() == {'a': bool, 'b': bool}

    def test_table_bool_infer_str_all_formats_from_schema(self):
        if False:
            i = 10
            return i + 15
        bool_data = [{'a': 'True', 'b': 'False'}, {'a': 't', 'b': 'f'}, {'a': 'true', 'b': 'false'}, {'a': 1, 'b': 0}, {'a': 'on', 'b': 'off'}]
        tbl = Table(bool_data)
        assert tbl.schema() == {'a': bool, 'b': bool}
        assert tbl.size() == 5
        assert tbl.view().to_dict() == {'a': [True, True, True, True, True], 'b': [False, False, False, False, False]}

    def test_table_infer_bool(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'a': [None, None, None, None, True, True, True]}
        tbl = Table(data)
        assert tbl.schema() == {'a': bool}

    def test_table_infer_str(self):
        if False:
            while True:
                i = 10
        data = {'a': [None, None, None, None, None, None, 'abc']}
        tbl = Table(data)
        assert tbl.schema() == {'a': str}

    def test_table_infer_time_as_string(self):
        if False:
            while True:
                i = 10
        data = {'a': [None, None, None, None, None, None, datetime(2019, 7, 11, 12, 30, 5).time()]}
        tbl = Table(data)
        assert tbl.schema() == {'a': str}

    def test_table_infer_date_from_datetime(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'a': [None, None, None, None, None, None, datetime(2019, 7, 11)]}
        tbl = Table(data)
        assert tbl.schema() == {'a': datetime}

    def test_table_infer_date_from_date(self):
        if False:
            i = 10
            return i + 15
        data = {'a': [None, None, None, None, None, None, date(2019, 7, 11)]}
        tbl = Table(data)
        assert tbl.schema() == {'a': date}

    def test_table_infer_valid_date(self):
        if False:
            return 10
        data = {'a': [None, None, None, None, None, None, '08/31/2019']}
        tbl = Table(data)
        assert tbl.schema() == {'a': date}

    def test_table_infer_ambiguous_date(self):
        if False:
            print('Hello World!')
        data = {'a': [None, None, None, None, None, None, '01/03/2019']}
        tbl = Table(data)
        assert tbl.schema() == {'a': date}

    def test_table_infer_ymd_date(self):
        if False:
            while True:
                i = 10
        data = {'a': [None, None, None, None, None, None, '2019/01/03']}
        tbl = Table(data)
        assert tbl.schema() == {'a': date}

    def test_table_infer_invalid_date(self):
        if False:
            return 10
        data = {'a': [None, None, None, None, None, None, '08/55/2019']}
        tbl = Table(data)
        assert tbl.schema() == {'a': str}

    def test_table_infer_date_edge(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'a': [None, None, None, None, None, None, '08/31/2019 00:00:00']}
        tbl = Table(data)
        assert tbl.schema() == {'a': date}

    def test_table_infer_datetime_edge(self):
        if False:
            while True:
                i = 10
        data = {'a': [None, None, None, None, None, None, '08/31/2019 00:00:01']}
        tbl = Table(data)
        assert tbl.schema() == {'a': datetime}

    def test_table_infer_valid_datetime(self):
        if False:
            return 10
        data = {'a': [None, None, None, None, None, None, '08/31/2019 07:30:00']}
        tbl = Table(data)
        assert tbl.schema() == {'a': datetime}

    def test_table_infer_iso_datetime(self):
        if False:
            while True:
                i = 10
        data = {'a': [None, None, None, None, None, None, '2019/07/25T09:00:00']}
        tbl = Table(data)
        assert tbl.schema() == {'a': datetime}

    def test_table_infer_datetime_separators(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'a': [None, None, None, None, None, '2019-07-25T09:00:00', '2019/07/25T09:00:00']}
        tbl = Table(data)
        assert tbl.schema() == {'a': datetime}

    def test_table_infer_datetime_tz(self):
        if False:
            return 10
        data = {'a': [None, None, None, None, None, '2019-07-25T09:00:00-05:00']}
        tbl = Table(data)
        assert tbl.schema() == {'a': datetime}

    def test_table_infer_invalid_datetime(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'a': [None, None, None, None, None, None, '08/31/2019 25:30:00']}
        tbl = Table(data)
        assert tbl.schema() == {'a': str}

    def test_table_infer_mixed_date(self):
        if False:
            i = 10
            return i + 15
        data = {'a': [None, None, None, None, None, '08/11/2019']}
        tbl = Table(data)
        assert tbl.schema() == {'a': date}

    def test_table_infer_mixed_datetime(self):
        if False:
            i = 10
            return i + 15
        data = {'a': [None, None, None, None, None, '08/11/2019 13:14:15']}
        tbl = Table(data)
        assert tbl.schema() == {'a': datetime}

    def test_table_strict_datetime_infer(self):
        if False:
            i = 10
            return i + 15
        data = {'a': ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1']}
        tbl = Table(data)
        assert tbl.schema() == {'a': str}

    def test_table_strict_date_infer(self):
        if False:
            i = 10
            return i + 15
        data = {'a': ['2019 09 10']}
        tbl = Table(data)
        assert tbl.schema() == {'a': date}

    def test_table_strict_datetime_separator_infer(self):
        if False:
            print('Hello World!')
        data = {'a': ['2019-10-01 7:30']}
        tbl = Table(data)
        assert tbl.schema() == {'a': datetime}

    def test_table_datetime_infer_no_false_positive(self):
        if False:
            while True:
                i = 10
        data = {'a': [' . - / but clearly not a date']}
        tbl = Table(data)
        assert tbl.schema() == {'a': str}

    @mark.skip
    def test_table_datetime_infer_from_string_with_time(self):
        if False:
            i = 10
            return i + 15
        data = {'a': ['11:00 ABCD']}
        tbl = Table(data)
        assert tbl.schema() == {'a': str}