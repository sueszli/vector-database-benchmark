"""Tests for Table Schema integration."""
from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, PeriodDtype
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import as_json_table_type, build_table_schema, convert_json_field_to_pandas_type, convert_pandas_type_to_json_field, set_default_names

@pytest.fixture
def df_schema():
    if False:
        i = 10
        return i + 15
    return DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'c'], 'C': pd.date_range('2016-01-01', freq='d', periods=4), 'D': pd.timedelta_range('1h', periods=4, freq='min')}, index=pd.Index(range(4), name='idx'))

@pytest.fixture
def df_table():
    if False:
        print('Hello World!')
    return DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'c'], 'C': pd.date_range('2016-01-01', freq='d', periods=4), 'D': pd.timedelta_range('1h', periods=4, freq='min'), 'E': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'])), 'F': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'], ordered=True)), 'G': [1.0, 2.0, 3, 4.0], 'H': pd.date_range('2016-01-01', freq='d', periods=4, tz='US/Central')}, index=pd.Index(range(4), name='idx'))

class TestBuildSchema:

    def test_build_table_schema(self, df_schema):
        if False:
            while True:
                i = 10
        result = build_table_schema(df_schema, version=False)
        expected = {'fields': [{'name': 'idx', 'type': 'integer'}, {'name': 'A', 'type': 'integer'}, {'name': 'B', 'type': 'string'}, {'name': 'C', 'type': 'datetime'}, {'name': 'D', 'type': 'duration'}], 'primaryKey': ['idx']}
        assert result == expected
        result = build_table_schema(df_schema)
        assert 'pandas_version' in result

    def test_series(self):
        if False:
            while True:
                i = 10
        s = pd.Series([1, 2, 3], name='foo')
        result = build_table_schema(s, version=False)
        expected = {'fields': [{'name': 'index', 'type': 'integer'}, {'name': 'foo', 'type': 'integer'}], 'primaryKey': ['index']}
        assert result == expected
        result = build_table_schema(s)
        assert 'pandas_version' in result

    def test_series_unnamed(self):
        if False:
            for i in range(10):
                print('nop')
        result = build_table_schema(pd.Series([1, 2, 3]), version=False)
        expected = {'fields': [{'name': 'index', 'type': 'integer'}, {'name': 'values', 'type': 'integer'}], 'primaryKey': ['index']}
        assert result == expected

    def test_multiindex(self, df_schema):
        if False:
            print('Hello World!')
        df = df_schema
        idx = pd.MultiIndex.from_product([('a', 'b'), (1, 2)])
        df.index = idx
        result = build_table_schema(df, version=False)
        expected = {'fields': [{'name': 'level_0', 'type': 'string'}, {'name': 'level_1', 'type': 'integer'}, {'name': 'A', 'type': 'integer'}, {'name': 'B', 'type': 'string'}, {'name': 'C', 'type': 'datetime'}, {'name': 'D', 'type': 'duration'}], 'primaryKey': ['level_0', 'level_1']}
        assert result == expected
        df.index.names = ['idx0', None]
        expected['fields'][0]['name'] = 'idx0'
        expected['primaryKey'] = ['idx0', 'level_1']
        result = build_table_schema(df, version=False)
        assert result == expected

class TestTableSchemaType:

    @pytest.mark.parametrize('int_type', [int, np.int16, np.int32, np.int64])
    def test_as_json_table_type_int_data(self, int_type):
        if False:
            while True:
                i = 10
        int_data = [1, 2, 3]
        assert as_json_table_type(np.array(int_data, dtype=int_type).dtype) == 'integer'

    @pytest.mark.parametrize('float_type', [float, np.float16, np.float32, np.float64])
    def test_as_json_table_type_float_data(self, float_type):
        if False:
            for i in range(10):
                print('nop')
        float_data = [1.0, 2.0, 3.0]
        assert as_json_table_type(np.array(float_data, dtype=float_type).dtype) == 'number'

    @pytest.mark.parametrize('bool_type', [bool, np.bool_])
    def test_as_json_table_type_bool_data(self, bool_type):
        if False:
            for i in range(10):
                print('nop')
        bool_data = [True, False]
        assert as_json_table_type(np.array(bool_data, dtype=bool_type).dtype) == 'boolean'

    @pytest.mark.parametrize('date_data', [pd.to_datetime(['2016']), pd.to_datetime(['2016'], utc=True), pd.Series(pd.to_datetime(['2016'])), pd.Series(pd.to_datetime(['2016'], utc=True)), pd.period_range('2016', freq='Y', periods=3)])
    def test_as_json_table_type_date_data(self, date_data):
        if False:
            i = 10
            return i + 15
        assert as_json_table_type(date_data.dtype) == 'datetime'

    @pytest.mark.parametrize('str_data', [pd.Series(['a', 'b']), pd.Index(['a', 'b'])])
    def test_as_json_table_type_string_data(self, str_data):
        if False:
            return 10
        assert as_json_table_type(str_data.dtype) == 'string'

    @pytest.mark.parametrize('cat_data', [pd.Categorical(['a']), pd.Categorical([1]), pd.Series(pd.Categorical([1])), pd.CategoricalIndex([1]), pd.Categorical([1])])
    def test_as_json_table_type_categorical_data(self, cat_data):
        if False:
            return 10
        assert as_json_table_type(cat_data.dtype) == 'any'

    @pytest.mark.parametrize('int_dtype', [int, np.int16, np.int32, np.int64])
    def test_as_json_table_type_int_dtypes(self, int_dtype):
        if False:
            i = 10
            return i + 15
        assert as_json_table_type(int_dtype) == 'integer'

    @pytest.mark.parametrize('float_dtype', [float, np.float16, np.float32, np.float64])
    def test_as_json_table_type_float_dtypes(self, float_dtype):
        if False:
            i = 10
            return i + 15
        assert as_json_table_type(float_dtype) == 'number'

    @pytest.mark.parametrize('bool_dtype', [bool, np.bool_])
    def test_as_json_table_type_bool_dtypes(self, bool_dtype):
        if False:
            for i in range(10):
                print('nop')
        assert as_json_table_type(bool_dtype) == 'boolean'

    @pytest.mark.parametrize('date_dtype', [np.dtype('<M8[ns]'), PeriodDtype('D'), DatetimeTZDtype('ns', 'US/Central')])
    def test_as_json_table_type_date_dtypes(self, date_dtype):
        if False:
            i = 10
            return i + 15
        assert as_json_table_type(date_dtype) == 'datetime'

    @pytest.mark.parametrize('td_dtype', [np.dtype('<m8[ns]')])
    def test_as_json_table_type_timedelta_dtypes(self, td_dtype):
        if False:
            i = 10
            return i + 15
        assert as_json_table_type(td_dtype) == 'duration'

    @pytest.mark.parametrize('str_dtype', [object])
    def test_as_json_table_type_string_dtypes(self, str_dtype):
        if False:
            print('Hello World!')
        assert as_json_table_type(str_dtype) == 'string'

    def test_as_json_table_type_categorical_dtypes(self):
        if False:
            while True:
                i = 10
        assert as_json_table_type(pd.Categorical(['a']).dtype) == 'any'
        assert as_json_table_type(CategoricalDtype()) == 'any'

class TestTableOrient:

    def test_build_series(self):
        if False:
            return 10
        s = pd.Series([1, 2], name='a')
        s.index.name = 'id'
        result = s.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result['schema']
        result['schema'].pop('pandas_version')
        fields = [{'name': 'id', 'type': 'integer'}, {'name': 'a', 'type': 'integer'}]
        schema = {'fields': fields, 'primaryKey': ['id']}
        expected = OrderedDict([('schema', schema), ('data', [OrderedDict([('id', 0), ('a', 1)]), OrderedDict([('id', 1), ('a', 2)])])])
        assert result == expected

    def test_read_json_from_to_json_results(self):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame({'_id': {'row_0': 0}, 'category': {'row_0': 'Goods'}, 'recommender_id': {'row_0': 3}, 'recommender_name_jp': {'row_0': '浦田'}, 'recommender_name_en': {'row_0': 'Urata'}, 'name_jp': {'row_0': '博多人形(松尾吉将まつお よしまさ)'}, 'name_en': {'row_0': 'Hakata Dolls Matsuo'}})
        result1 = pd.read_json(StringIO(df.to_json()))
        result2 = DataFrame.from_dict(json.loads(df.to_json()))
        tm.assert_frame_equal(result1, df)
        tm.assert_frame_equal(result2, df)

    def test_to_json(self, df_table):
        if False:
            for i in range(10):
                print('nop')
        df = df_table
        df.index.name = 'idx'
        result = df.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result['schema']
        result['schema'].pop('pandas_version')
        fields = [{'name': 'idx', 'type': 'integer'}, {'name': 'A', 'type': 'integer'}, {'name': 'B', 'type': 'string'}, {'name': 'C', 'type': 'datetime'}, {'name': 'D', 'type': 'duration'}, {'constraints': {'enum': ['a', 'b', 'c']}, 'name': 'E', 'ordered': False, 'type': 'any'}, {'constraints': {'enum': ['a', 'b', 'c']}, 'name': 'F', 'ordered': True, 'type': 'any'}, {'name': 'G', 'type': 'number'}, {'name': 'H', 'type': 'datetime', 'tz': 'US/Central'}]
        schema = {'fields': fields, 'primaryKey': ['idx']}
        data = [OrderedDict([('idx', 0), ('A', 1), ('B', 'a'), ('C', '2016-01-01T00:00:00.000'), ('D', 'P0DT1H0M0S'), ('E', 'a'), ('F', 'a'), ('G', 1.0), ('H', '2016-01-01T06:00:00.000Z')]), OrderedDict([('idx', 1), ('A', 2), ('B', 'b'), ('C', '2016-01-02T00:00:00.000'), ('D', 'P0DT1H1M0S'), ('E', 'b'), ('F', 'b'), ('G', 2.0), ('H', '2016-01-02T06:00:00.000Z')]), OrderedDict([('idx', 2), ('A', 3), ('B', 'c'), ('C', '2016-01-03T00:00:00.000'), ('D', 'P0DT1H2M0S'), ('E', 'c'), ('F', 'c'), ('G', 3.0), ('H', '2016-01-03T06:00:00.000Z')]), OrderedDict([('idx', 3), ('A', 4), ('B', 'c'), ('C', '2016-01-04T00:00:00.000'), ('D', 'P0DT1H3M0S'), ('E', 'c'), ('F', 'c'), ('G', 4.0), ('H', '2016-01-04T06:00:00.000Z')])]
        expected = OrderedDict([('schema', schema), ('data', data)])
        assert result == expected

    def test_to_json_float_index(self):
        if False:
            while True:
                i = 10
        data = pd.Series(1, index=[1.0, 2.0])
        result = data.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result['schema'].pop('pandas_version')
        expected = OrderedDict([('schema', {'fields': [{'name': 'index', 'type': 'number'}, {'name': 'values', 'type': 'integer'}], 'primaryKey': ['index']}), ('data', [OrderedDict([('index', 1.0), ('values', 1)]), OrderedDict([('index', 2.0), ('values', 1)])])])
        assert result == expected

    def test_to_json_period_index(self):
        if False:
            return 10
        idx = pd.period_range('2016', freq='Q-JAN', periods=2)
        data = pd.Series(1, idx)
        result = data.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result['schema'].pop('pandas_version')
        fields = [{'freq': 'QE-JAN', 'name': 'index', 'type': 'datetime'}, {'name': 'values', 'type': 'integer'}]
        schema = {'fields': fields, 'primaryKey': ['index']}
        data = [OrderedDict([('index', '2015-11-01T00:00:00.000'), ('values', 1)]), OrderedDict([('index', '2016-02-01T00:00:00.000'), ('values', 1)])]
        expected = OrderedDict([('schema', schema), ('data', data)])
        assert result == expected

    def test_to_json_categorical_index(self):
        if False:
            i = 10
            return i + 15
        data = pd.Series(1, pd.CategoricalIndex(['a', 'b']))
        result = data.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result['schema'].pop('pandas_version')
        expected = OrderedDict([('schema', {'fields': [{'name': 'index', 'type': 'any', 'constraints': {'enum': ['a', 'b']}, 'ordered': False}, {'name': 'values', 'type': 'integer'}], 'primaryKey': ['index']}), ('data', [OrderedDict([('index', 'a'), ('values', 1)]), OrderedDict([('index', 'b'), ('values', 1)])])])
        assert result == expected

    def test_date_format_raises(self, df_table):
        if False:
            return 10
        msg = "Trying to write with `orient='table'` and `date_format='epoch'`. Table Schema requires dates to be formatted with `date_format='iso'`"
        with pytest.raises(ValueError, match=msg):
            df_table.to_json(orient='table', date_format='epoch')
        df_table.to_json(orient='table', date_format='iso')
        df_table.to_json(orient='table')

    def test_convert_pandas_type_to_json_field_int(self, index_or_series):
        if False:
            while True:
                i = 10
        kind = index_or_series
        data = [1, 2, 3]
        result = convert_pandas_type_to_json_field(kind(data, name='name'))
        expected = {'name': 'name', 'type': 'integer'}
        assert result == expected

    def test_convert_pandas_type_to_json_field_float(self, index_or_series):
        if False:
            i = 10
            return i + 15
        kind = index_or_series
        data = [1.0, 2.0, 3.0]
        result = convert_pandas_type_to_json_field(kind(data, name='name'))
        expected = {'name': 'name', 'type': 'number'}
        assert result == expected

    @pytest.mark.parametrize('dt_args,extra_exp', [({}, {}), ({'utc': True}, {'tz': 'UTC'})])
    @pytest.mark.parametrize('wrapper', [None, pd.Series])
    def test_convert_pandas_type_to_json_field_datetime(self, dt_args, extra_exp, wrapper):
        if False:
            print('Hello World!')
        data = [1.0, 2.0, 3.0]
        data = pd.to_datetime(data, **dt_args)
        if wrapper is pd.Series:
            data = pd.Series(data, name='values')
        result = convert_pandas_type_to_json_field(data)
        expected = {'name': 'values', 'type': 'datetime'}
        expected.update(extra_exp)
        assert result == expected

    def test_convert_pandas_type_to_json_period_range(self):
        if False:
            print('Hello World!')
        arr = pd.period_range('2016', freq='Y-DEC', periods=4)
        result = convert_pandas_type_to_json_field(arr)
        expected = {'name': 'values', 'type': 'datetime', 'freq': 'YE-DEC'}
        assert result == expected

    @pytest.mark.parametrize('kind', [pd.Categorical, pd.CategoricalIndex])
    @pytest.mark.parametrize('ordered', [True, False])
    def test_convert_pandas_type_to_json_field_categorical(self, kind, ordered):
        if False:
            i = 10
            return i + 15
        data = ['a', 'b', 'c']
        if kind is pd.Categorical:
            arr = pd.Series(kind(data, ordered=ordered), name='cats')
        elif kind is pd.CategoricalIndex:
            arr = kind(data, ordered=ordered, name='cats')
        result = convert_pandas_type_to_json_field(arr)
        expected = {'name': 'cats', 'type': 'any', 'constraints': {'enum': data}, 'ordered': ordered}
        assert result == expected

    @pytest.mark.parametrize('inp,exp', [({'type': 'integer'}, 'int64'), ({'type': 'number'}, 'float64'), ({'type': 'boolean'}, 'bool'), ({'type': 'duration'}, 'timedelta64'), ({'type': 'datetime'}, 'datetime64[ns]'), ({'type': 'datetime', 'tz': 'US/Hawaii'}, 'datetime64[ns, US/Hawaii]'), ({'type': 'any'}, 'object'), ({'type': 'any', 'constraints': {'enum': ['a', 'b', 'c']}, 'ordered': False}, CategoricalDtype(categories=['a', 'b', 'c'], ordered=False)), ({'type': 'any', 'constraints': {'enum': ['a', 'b', 'c']}, 'ordered': True}, CategoricalDtype(categories=['a', 'b', 'c'], ordered=True)), ({'type': 'string'}, 'object')])
    def test_convert_json_field_to_pandas_type(self, inp, exp):
        if False:
            return 10
        field = {'name': 'foo'}
        field.update(inp)
        assert convert_json_field_to_pandas_type(field) == exp

    @pytest.mark.parametrize('inp', ['geopoint', 'geojson', 'fake_type'])
    def test_convert_json_field_to_pandas_type_raises(self, inp):
        if False:
            i = 10
            return i + 15
        field = {'type': inp}
        with pytest.raises(ValueError, match=f'Unsupported or invalid field type: {inp}'):
            convert_json_field_to_pandas_type(field)

    def test_categorical(self):
        if False:
            print('Hello World!')
        s = pd.Series(pd.Categorical(['a', 'b', 'a']))
        s.index.name = 'idx'
        result = s.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result['schema'].pop('pandas_version')
        fields = [{'name': 'idx', 'type': 'integer'}, {'constraints': {'enum': ['a', 'b']}, 'name': 'values', 'ordered': False, 'type': 'any'}]
        expected = OrderedDict([('schema', {'fields': fields, 'primaryKey': ['idx']}), ('data', [OrderedDict([('idx', 0), ('values', 'a')]), OrderedDict([('idx', 1), ('values', 'b')]), OrderedDict([('idx', 2), ('values', 'a')])])])
        assert result == expected

    @pytest.mark.parametrize('idx,nm,prop', [(pd.Index([1]), 'index', 'name'), (pd.Index([1], name='myname'), 'myname', 'name'), (pd.MultiIndex.from_product([('a', 'b'), ('c', 'd')]), ['level_0', 'level_1'], 'names'), (pd.MultiIndex.from_product([('a', 'b'), ('c', 'd')], names=['n1', 'n2']), ['n1', 'n2'], 'names'), (pd.MultiIndex.from_product([('a', 'b'), ('c', 'd')], names=['n1', None]), ['n1', 'level_1'], 'names')])
    def test_set_names_unset(self, idx, nm, prop):
        if False:
            return 10
        data = pd.Series(1, idx)
        result = set_default_names(data)
        assert getattr(result.index, prop) == nm

    @pytest.mark.parametrize('idx', [pd.Index([], name='index'), pd.MultiIndex.from_arrays([['foo'], ['bar']], names=('level_0', 'level_1')), pd.MultiIndex.from_arrays([['foo'], ['bar']], names=('foo', 'level_1'))])
    def test_warns_non_roundtrippable_names(self, idx):
        if False:
            i = 10
            return i + 15
        df = DataFrame(index=idx)
        df.index.name = 'index'
        with tm.assert_produces_warning():
            set_default_names(df)

    def test_timestamp_in_columns(self):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame([[1, 2]], columns=[pd.Timestamp('2016'), pd.Timedelta(10, unit='s')])
        result = df.to_json(orient='table')
        js = json.loads(result)
        assert js['schema']['fields'][1]['name'] == '2016-01-01T00:00:00.000'
        assert js['schema']['fields'][2]['name'] == 'P0DT0H0M10S'

    @pytest.mark.parametrize('case', [pd.Series([1], index=pd.Index([1], name='a'), name='a'), DataFrame({'A': [1]}, index=pd.Index([1], name='A')), DataFrame({'A': [1]}, index=pd.MultiIndex.from_arrays([['a'], [1]], names=['A', 'a']))])
    def test_overlapping_names(self, case):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match='Overlapping'):
            case.to_json(orient='table')

    def test_mi_falsey_name(self):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=pd.MultiIndex.from_product([('A', 'B'), ('a', 'b')]))
        result = [x['name'] for x in build_table_schema(df)['fields']]
        assert result == ['level_0', 'level_1', 0, 1, 2, 3]

class TestTableOrientReader:

    @pytest.mark.parametrize('index_nm', [None, 'idx', pytest.param('index', marks=pytest.mark.xfail), 'level_0'])
    @pytest.mark.parametrize('vals', [{'ints': [1, 2, 3, 4]}, {'objects': ['a', 'b', 'c', 'd']}, {'objects': ['1', '2', '3', '4']}, {'date_ranges': pd.date_range('2016-01-01', freq='d', periods=4)}, {'categoricals': pd.Series(pd.Categorical(['a', 'b', 'c', 'c']))}, {'ordered_cats': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'], ordered=True))}, {'floats': [1.0, 2.0, 3.0, 4.0]}, {'floats': [1.1, 2.2, 3.3, 4.4]}, {'bools': [True, False, False, True]}, {'timezones': pd.date_range('2016-01-01', freq='d', periods=4, tz='US/Central')}])
    def test_read_json_table_orient(self, index_nm, vals, recwarn):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        out = df.to_json(orient='table')
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize('index_nm', [None, 'idx', 'index'])
    @pytest.mark.parametrize('vals', [{'timedeltas': pd.timedelta_range('1h', periods=4, freq='min')}])
    def test_read_json_table_orient_raises(self, index_nm, vals, recwarn):
        if False:
            return 10
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        out = df.to_json(orient='table')
        with pytest.raises(NotImplementedError, match='can not yet read '):
            pd.read_json(out, orient='table')

    @pytest.mark.parametrize('index_nm', [None, 'idx', pytest.param('index', marks=pytest.mark.xfail), 'level_0'])
    @pytest.mark.parametrize('vals', [{'ints': [1, 2, 3, 4]}, {'objects': ['a', 'b', 'c', 'd']}, {'objects': ['1', '2', '3', '4']}, {'date_ranges': pd.date_range('2016-01-01', freq='d', periods=4)}, {'categoricals': pd.Series(pd.Categorical(['a', 'b', 'c', 'c']))}, {'ordered_cats': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'], ordered=True))}, {'floats': [1.0, 2.0, 3.0, 4.0]}, {'floats': [1.1, 2.2, 3.3, 4.4]}, {'bools': [True, False, False, True]}, {'timezones': pd.date_range('2016-01-01', freq='d', periods=4, tz='US/Central')}])
    def test_read_json_table_period_orient(self, index_nm, vals, recwarn):
        if False:
            i = 10
            return i + 15
        df = DataFrame(vals, index=pd.Index((pd.Period(f'2022Q{q}') for q in range(1, 5)), name=index_nm))
        out = df.to_json(orient='table')
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize('idx', [pd.Index(range(4)), pd.date_range('2020-08-30', freq='d', periods=4)._with_freq(None), pd.date_range('2020-08-30', freq='d', periods=4, tz='US/Central')._with_freq(None), pd.MultiIndex.from_product([pd.date_range('2020-08-30', freq='d', periods=2, tz='US/Central'), ['x', 'y']])])
    @pytest.mark.parametrize('vals', [{'floats': [1.1, 2.2, 3.3, 4.4]}, {'dates': pd.date_range('2020-08-30', freq='d', periods=4)}, {'timezones': pd.date_range('2020-08-30', freq='d', periods=4, tz='Europe/London')}])
    def test_read_json_table_timezones_orient(self, idx, vals, recwarn):
        if False:
            return 10
        df = DataFrame(vals, index=idx)
        out = df.to_json(orient='table')
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    def test_comprehensive(self):
        if False:
            print('Hello World!')
        df = DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'c'], 'C': pd.date_range('2016-01-01', freq='d', periods=4), 'E': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'])), 'F': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'], ordered=True)), 'G': [1.1, 2.2, 3.3, 4.4], 'H': pd.date_range('2016-01-01', freq='d', periods=4, tz='US/Central'), 'I': [True, False, False, True]}, index=pd.Index(range(4), name='idx'))
        out = StringIO(df.to_json(orient='table'))
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize('index_names', [[None, None], ['foo', 'bar'], ['foo', None], [None, 'foo'], ['index', 'foo']])
    def test_multiindex(self, index_names):
        if False:
            while True:
                i = 10
        df = DataFrame([['Arr', 'alpha', [1, 2, 3, 4]], ['Bee', 'Beta', [10, 20, 30, 40]]], index=[['A', 'B'], ['Null', 'Eins']], columns=['Aussprache', 'Griechisch', 'Args'])
        df.index.names = index_names
        out = StringIO(df.to_json(orient='table'))
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    def test_empty_frame_roundtrip(self):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(columns=['a', 'b', 'c'])
        expected = df.copy()
        out = StringIO(df.to_json(orient='table'))
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(expected, result)

    def test_read_json_orient_table_old_schema_version(self):
        if False:
            print('Hello World!')
        df_json = '\n        {\n            "schema":{\n                "fields":[\n                    {"name":"index","type":"integer"},\n                    {"name":"a","type":"string"}\n                ],\n                "primaryKey":["index"],\n                "pandas_version":"0.20.0"\n            },\n            "data":[\n                {"index":0,"a":1},\n                {"index":1,"a":2.0},\n                {"index":2,"a":"s"}\n            ]\n        }\n        '
        expected = DataFrame({'a': [1, 2.0, 's']})
        result = pd.read_json(StringIO(df_json), orient='table')
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize('freq', ['M', '2M', 'Q', '2Q', 'Y', '2Y'])
    def test_read_json_table_orient_period_depr_freq(self, freq, recwarn):
        if False:
            print('Hello World!')
        df = DataFrame({'ints': [1, 2]}, index=pd.PeriodIndex(['2020-01', '2021-06'], freq=freq))
        out = df.to_json(orient='table')
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)