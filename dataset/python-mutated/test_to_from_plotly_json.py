import pytest
import plotly.io.json as pio
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
import datetime
import re
import sys
import warnings
from pytz import timezone
from _plotly_utils.optional_imports import get_module
orjson = get_module('orjson')
eastern = timezone('US/Eastern')

def build_json_opts(pretty=False):
    if False:
        print('Hello World!')
    opts = {'sort_keys': True}
    if pretty:
        opts['indent'] = 2
    else:
        opts['separators'] = (',', ':')
    return opts

def to_json_test(value, pretty=False):
    if False:
        for i in range(10):
            print('nop')
    return json.dumps(value, **build_json_opts(pretty=pretty))

def isoformat_test(dt_value):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(dt_value, np.datetime64):
        return str(dt_value)
    elif isinstance(dt_value, datetime.datetime):
        return dt_value.isoformat()
    else:
        raise ValueError('Unsupported date type: {}'.format(type(dt_value)))

def build_test_dict(value):
    if False:
        for i in range(10):
            print('nop')
    return dict(a=value, b=[3, value], c=dict(Z=value))

def build_test_dict_string(value_string, pretty=False):
    if False:
        while True:
            i = 10
    if pretty:
        non_pretty_str = build_test_dict_string(value_string, pretty=False)
        return to_json_test(json.loads(non_pretty_str), pretty=True)
    else:
        value_string = str(value_string).replace(' ', '')
        return '{"a":%s,"b":[3,%s],"c":{"Z":%s}}' % tuple([value_string] * 3)

def check_roundtrip(value, engine, pretty):
    if False:
        while True:
            i = 10
    encoded = pio.to_json_plotly(value, engine=engine, pretty=pretty)
    decoded = pio.from_json_plotly(encoded, engine=engine)
    reencoded = pio.to_json_plotly(decoded, engine=engine, pretty=pretty)
    assert encoded == reencoded
    if sys.version_info.major == 3:
        encoded_bytes = encoded.encode('utf8')
        decoded_from_bytes = pio.from_json_plotly(encoded_bytes, engine=engine)
        assert decoded == decoded_from_bytes
if orjson is not None:
    engines = ['json', 'orjson', 'auto']
else:
    engines = ['json', 'auto']

@pytest.fixture(scope='module', params=engines)
def engine(request):
    if False:
        i = 10
        return i + 15
    return request.param

@pytest.fixture(scope='module', params=[False])
def pretty(request):
    if False:
        return 10
    return request.param

@pytest.fixture(scope='module', params=['float64', 'int32', 'uint32'])
def graph_object(request):
    if False:
        return 10
    return request.param

@pytest.fixture(scope='module', params=['float64', 'int32', 'uint32'])
def numeric_numpy_array(request):
    if False:
        i = 10
        return i + 15
    dtype = request.param
    return np.linspace(-5, 5, 4, dtype=dtype)

@pytest.fixture(scope='module')
def object_numpy_array(request):
    if False:
        return 10
    return np.array(['a', 1, [2, 3]])

@pytest.fixture(scope='module')
def numpy_unicode_array(request):
    if False:
        i = 10
        return i + 15
    return np.array(['A', 'BB', 'CCC'], dtype='U')

@pytest.fixture(scope='module', params=[datetime.datetime(2003, 7, 12, 8, 34, 22), datetime.datetime.now(), np.datetime64(datetime.datetime.utcnow()), pd.Timestamp(datetime.datetime.now()), eastern.localize(datetime.datetime(2003, 7, 12, 8, 34, 22)), eastern.localize(datetime.datetime.now()), pd.Timestamp(datetime.datetime.now(), tzinfo=eastern)])
def datetime_value(request):
    if False:
        while True:
            i = 10
    return request.param

@pytest.fixture(params=[list, lambda a: pd.DatetimeIndex(a), lambda a: pd.Series(pd.DatetimeIndex(a)), lambda a: pd.DatetimeIndex(a).values, lambda a: np.array(a, dtype='object')])
def datetime_array(request, datetime_value):
    if False:
        i = 10
        return i + 15
    return request.param([datetime_value] * 3)

def test_graph_object_input(engine, pretty):
    if False:
        i = 10
        return i + 15
    scatter = go.Scatter(x=[1, 2, 3], y=np.array([4, 5, 6]))
    result = pio.to_json_plotly(scatter, engine=engine)
    expected = '{"x":[1,2,3],"y":[4,5,6],"type":"scatter"}'
    assert result == expected
    check_roundtrip(result, engine=engine, pretty=pretty)

def test_numeric_numpy_encoding(numeric_numpy_array, engine, pretty):
    if False:
        for i in range(10):
            print('nop')
    value = build_test_dict(numeric_numpy_array)
    result = pio.to_json_plotly(value, engine=engine, pretty=pretty)
    array_str = to_json_test(numeric_numpy_array.tolist())
    expected = build_test_dict_string(array_str, pretty=pretty)
    assert result == expected
    check_roundtrip(result, engine=engine, pretty=pretty)

def test_numpy_unicode_encoding(numpy_unicode_array, engine, pretty):
    if False:
        print('Hello World!')
    value = build_test_dict(numpy_unicode_array)
    result = pio.to_json_plotly(value, engine=engine, pretty=pretty)
    array_str = to_json_test(numpy_unicode_array.tolist())
    expected = build_test_dict_string(array_str)
    assert result == expected
    check_roundtrip(result, engine=engine, pretty=pretty)

def test_object_numpy_encoding(object_numpy_array, engine, pretty):
    if False:
        return 10
    value = build_test_dict(object_numpy_array)
    result = pio.to_json_plotly(value, engine=engine, pretty=pretty)
    array_str = to_json_test(object_numpy_array.tolist())
    expected = build_test_dict_string(array_str)
    assert result == expected
    check_roundtrip(result, engine=engine, pretty=pretty)

def test_datetime(datetime_value, engine, pretty):
    if False:
        for i in range(10):
            print('nop')
    value = build_test_dict(datetime_value)
    result = pio.to_json_plotly(value, engine=engine, pretty=pretty)
    expected = build_test_dict_string('"{}"'.format(isoformat_test(datetime_value)))
    assert result == expected
    check_roundtrip(result, engine=engine, pretty=pretty)

def test_datetime_arrays(datetime_array, engine, pretty):
    if False:
        print('Hello World!')
    value = build_test_dict(datetime_array)
    result = pio.to_json_plotly(value, engine=engine)

    def to_str(v):
        if False:
            print('Hello World!')
        try:
            v = v.isoformat(sep='T')
        except (TypeError, AttributeError):
            pass
        return str(v)
    if isinstance(datetime_array, list):
        dt_values = [to_str(d) for d in datetime_array]
    elif isinstance(datetime_array, pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            dt_values = [to_str(d) for d in np.array(datetime_array.dt.to_pydatetime()).tolist()]
    elif isinstance(datetime_array, pd.DatetimeIndex):
        dt_values = [to_str(d) for d in datetime_array.to_pydatetime().tolist()]
    else:
        dt_values = [to_str(d) for d in datetime_array]
    array_str = to_json_test(dt_values)
    expected = build_test_dict_string(array_str)
    if orjson:
        trailing_zeros = re.compile('[.]?0+"')
        result = trailing_zeros.sub('"', result)
        expected = trailing_zeros.sub('"', expected)
    assert result == expected
    check_roundtrip(result, engine=engine, pretty=pretty)

def test_object_array(engine, pretty):
    if False:
        while True:
            i = 10
    fig = px.scatter(px.data.tips(), x='total_bill', y='tip', custom_data=['sex'])
    result = fig.to_plotly_json()
    check_roundtrip(result, engine=engine, pretty=pretty)

def test_nonstring_key(engine, pretty):
    if False:
        print('Hello World!')
    value = build_test_dict({0: 1})
    result = pio.to_json_plotly(value, engine=engine)
    check_roundtrip(result, engine=engine, pretty=pretty)

def test_mixed_string_nonstring_key(engine, pretty):
    if False:
        for i in range(10):
            print('nop')
    value = build_test_dict({0: 1, 'a': 2})
    result = pio.to_json_plotly(value, engine=engine)
    check_roundtrip(result, engine=engine, pretty=pretty)

def test_sanitize_json(engine):
    if False:
        return 10
    layout = {'title': {'text': '</script>\u2028\u2029'}}
    fig = go.Figure(layout=layout)
    fig_json = pio.to_json_plotly(fig, engine=engine)
    layout_2 = json.loads(fig_json)['layout']
    del layout_2['template']
    assert layout == layout_2
    replacements = {'<': '\\u003c', '>': '\\u003e', '/': '\\u002f', '\u2028': '\\u2028', '\u2029': '\\u2029'}
    for (bad, good) in replacements.items():
        assert bad not in fig_json
        assert good in fig_json