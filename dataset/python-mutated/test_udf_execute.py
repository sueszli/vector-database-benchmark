from __future__ import annotations
import os
import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param
import ibis
import ibis.expr.datatypes as dt
from ibis.backends.bigquery import udf
PROJECT_ID = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID', 'ibis-gbq')
DATASET_ID = 'testing'

@pytest.fixture(scope='module')
def alltypes(con):
    if False:
        while True:
            i = 10
    t = con.table('functional_alltypes')
    expr = t[t.bigint_col.isin([10, 20])].limit(10)
    return expr

@pytest.fixture(scope='module')
def df(alltypes):
    if False:
        print('Hello World!')
    return alltypes.execute()

def test_udf(alltypes, df):
    if False:
        for i in range(10):
            print('nop')

    @udf(input_type=[dt.double, dt.double], output_type=dt.double, determinism=True)
    def my_add(a, b):
        if False:
            i = 10
            return i + 15
        return a + b
    expr = my_add(alltypes.double_col, alltypes.double_col)
    result = expr.execute()
    assert not result.empty
    expected = (df.double_col + df.double_col).rename('tmp')
    tm.assert_series_equal(result.value_counts().sort_index(), expected.value_counts().sort_index(), check_names=False)

def test_udf_with_struct(alltypes, df, snapshot):
    if False:
        i = 10
        return i + 15

    @udf(input_type=[dt.double, dt.double], output_type=dt.Struct.from_tuples([('width', dt.double), ('height', dt.double)]))
    def my_struct_thing(a, b):
        if False:
            while True:
                i = 10

        class Rectangle:

            def __init__(self, width, height):
                if False:
                    for i in range(10):
                        print('nop')
                self.width = width
                self.height = height
        return Rectangle(a, b)
    result = my_struct_thing.sql
    snapshot.assert_match(result, 'out.sql')
    expr = my_struct_thing(alltypes.double_col, alltypes.double_col)
    result = expr.execute()
    assert not result.empty
    expected = pd.Series([{'width': c, 'height': c} for c in df.double_col], name='tmp')
    tm.assert_series_equal(result, expected, check_names=False)

def test_udf_compose(alltypes, df):
    if False:
        print('Hello World!')

    @udf([dt.double], dt.double)
    def add_one(x):
        if False:
            while True:
                i = 10
        return x + 1.0

    @udf([dt.double], dt.double)
    def times_two(x):
        if False:
            while True:
                i = 10
        return x * 2.0
    t = alltypes
    expr = times_two(add_one(t.double_col))
    result = expr.execute()
    expected = ((df.double_col + 1.0) * 2.0).rename('tmp')
    tm.assert_series_equal(result, expected, check_names=False)

def test_udf_scalar(con):
    if False:
        i = 10
        return i + 15

    @udf([dt.double, dt.double], dt.double)
    def my_add(x, y):
        if False:
            while True:
                i = 10
        return x + y
    expr = my_add(1, 2)
    result = con.execute(expr)
    assert result == 3

def test_multiple_calls_has_one_definition(con):
    if False:
        print('Hello World!')

    @udf([dt.string], dt.double)
    def my_str_len(s):
        if False:
            i = 10
            return i + 15
        return s.length
    s = ibis.literal('abcd')
    expr = my_str_len(s) + my_str_len(s)
    add = expr.op()
    assert add.left.sql == add.right.sql
    assert con.execute(expr) == 8.0

def test_udf_libraries(con):
    if False:
        i = 10
        return i + 15

    @udf([dt.Array(dt.string)], dt.double, libraries=['gs://ibis-testing-libraries/lodash.min.js'])
    def string_length(strings):
        if False:
            return 10
        return _.sum(_.map(strings, lambda x: x.length))
    raw_data = ['aaa', 'bb', 'c']
    data = ibis.literal(raw_data)
    expr = string_length(data)
    result = con.execute(expr)
    expected = sum(map(len, raw_data))
    assert result == expected

def test_udf_with_len(con):
    if False:
        print('Hello World!')

    @udf([dt.string], dt.double)
    def my_str_len(x):
        if False:
            for i in range(10):
                print('nop')
        return len(x)

    @udf([dt.Array(dt.string)], dt.double)
    def my_array_len(x):
        if False:
            return 10
        return len(x)
    assert con.execute(my_str_len('aaa')) == 3
    assert con.execute(my_array_len(['aaa', 'bb'])) == 2

@pytest.mark.parametrize(('argument_type',), [param(dt.string, id='string'), param('ANY TYPE', id='string')])
def test_udf_sql(con, argument_type):
    if False:
        for i in range(10):
            print('nop')
    format_t = udf.sql('format_t', params={'input': argument_type}, output_type=dt.string, sql_expression="FORMAT('%T', input)")
    s = ibis.literal('abcd')
    expr = format_t(s)
    con.execute(expr)

@pytest.mark.parametrize(('value', 'expected'), [param(b'', 0, id='empty'), param(b'\x00', 0, id='zero'), param(b'\x05', 2, id='two'), param(b'\x00\x08', 1, id='one'), param(b'\xff\xff', 16, id='sixteen'), param(b'\xff\xff\xff\xff\xff\xff\xff\xfe', 63, id='sixty-three'), param(b'\xff\xff\xff\xff\xff\xff\xff\xff', 64, id='sixty-four'), param(b'\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff', 80, id='eighty')])
def test_builtin_scalar(con, value, expected):
    if False:
        return 10
    from ibis import udf

    @udf.scalar.builtin
    def bit_count(x: bytes) -> int:
        if False:
            while True:
                i = 10
        ...
    expr = bit_count(value)
    result = con.execute(expr)
    assert result == expected

@pytest.mark.parametrize(('where', 'expected'), [param({'where': True}, list('abcdef'), id='where-true'), param({'where': False}, [], id='where-false'), param({}, list('abcdef'), id='where-nothing')])
def test_builtin_agg(con, where, expected):
    if False:
        i = 10
        return i + 15
    from ibis import udf

    @udf.agg.builtin(name='array_concat_agg')
    def concat_agg(x, where: bool=True) -> dt.Array[str]:
        if False:
            while True:
                i = 10
        ...
    t = ibis.memtable({'a': [list('abc'), list('def')]})
    expr = concat_agg(t.a, **where)
    result = con.execute(expr)
    assert result == expected