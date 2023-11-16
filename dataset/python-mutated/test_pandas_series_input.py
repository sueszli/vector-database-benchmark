import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from _plotly_utils.basevalidators import NumberValidator, IntegerValidator, DataArrayValidator, ColorValidator

@pytest.fixture
def data_array_validator(request):
    if False:
        return 10
    return DataArrayValidator('prop', 'parent')

@pytest.fixture
def integer_validator(request):
    if False:
        while True:
            i = 10
    return IntegerValidator('prop', 'parent', array_ok=True)

@pytest.fixture
def number_validator(request):
    if False:
        print('Hello World!')
    return NumberValidator('prop', 'parent', array_ok=True)

@pytest.fixture
def color_validator(request):
    if False:
        for i in range(10):
            print('nop')
    return ColorValidator('prop', 'parent', array_ok=True, colorscale_path='')

@pytest.fixture(params=['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64'])
def numeric_dtype(request):
    if False:
        for i in range(10):
            print('nop')
    return request.param

@pytest.fixture(params=[pd.Series, pd.Index])
def pandas_type(request):
    if False:
        print('Hello World!')
    return request.param

@pytest.fixture
def numeric_pandas(request, pandas_type, numeric_dtype):
    if False:
        print('Hello World!')
    return pandas_type(np.arange(10), dtype=numeric_dtype)

@pytest.fixture
def color_object_pandas(request, pandas_type):
    if False:
        print('Hello World!')
    return pandas_type(['blue', 'green', 'red'] * 3, dtype='object')

@pytest.fixture
def color_categorical_pandas(request, pandas_type):
    if False:
        return 10
    return pandas_type(pd.Categorical(['blue', 'green', 'red'] * 3))

@pytest.fixture
def dates_array(request):
    if False:
        while True:
            i = 10
    return np.array([datetime(year=2013, month=10, day=10), datetime(year=2013, month=11, day=10), datetime(year=2013, month=12, day=10), datetime(year=2014, month=1, day=10), datetime(year=2014, month=2, day=10)])

@pytest.fixture
def datetime_pandas(request, pandas_type, dates_array):
    if False:
        print('Hello World!')
    return pandas_type(dates_array)

def test_numeric_validator_numeric_pandas(number_validator, numeric_pandas):
    if False:
        return 10
    res = number_validator.validate_coerce(numeric_pandas)
    assert isinstance(res, np.ndarray)
    assert res.dtype == numeric_pandas.dtype
    np.testing.assert_array_equal(res, numeric_pandas)

def test_integer_validator_numeric_pandas(integer_validator, numeric_pandas):
    if False:
        while True:
            i = 10
    res = integer_validator.validate_coerce(numeric_pandas)
    assert isinstance(res, np.ndarray)
    if numeric_pandas.dtype.kind in ('u', 'i'):
        assert res.dtype == numeric_pandas.dtype
    else:
        assert res.dtype == 'int32'
    np.testing.assert_array_equal(res, numeric_pandas)

def test_data_array_validator(data_array_validator, numeric_pandas):
    if False:
        while True:
            i = 10
    res = data_array_validator.validate_coerce(numeric_pandas)
    assert isinstance(res, np.ndarray)
    assert res.dtype == numeric_pandas.dtype
    np.testing.assert_array_equal(res, numeric_pandas)

def test_color_validator_numeric(color_validator, numeric_pandas):
    if False:
        for i in range(10):
            print('nop')
    res = color_validator.validate_coerce(numeric_pandas)
    assert isinstance(res, np.ndarray)
    assert res.dtype == numeric_pandas.dtype
    np.testing.assert_array_equal(res, numeric_pandas)

def test_color_validator_object(color_validator, color_object_pandas):
    if False:
        i = 10
        return i + 15
    res = color_validator.validate_coerce(color_object_pandas)
    assert isinstance(res, np.ndarray)
    assert res.dtype == 'object'
    np.testing.assert_array_equal(res, color_object_pandas)

def test_color_validator_categorical(color_validator, color_categorical_pandas):
    if False:
        print('Hello World!')
    res = color_validator.validate_coerce(color_categorical_pandas)
    assert color_categorical_pandas.dtype == 'category'
    assert isinstance(res, np.ndarray)
    assert res.dtype == 'object'
    np.testing.assert_array_equal(res, np.array(color_categorical_pandas))

def test_data_array_validator_dates_series(data_array_validator, datetime_pandas, dates_array):
    if False:
        return 10
    res = data_array_validator.validate_coerce(datetime_pandas)
    assert isinstance(res, np.ndarray)
    assert res.dtype == 'object'
    np.testing.assert_array_equal(res, dates_array)

def test_data_array_validator_dates_dataframe(data_array_validator, datetime_pandas, dates_array):
    if False:
        print('Hello World!')
    df = pd.DataFrame({'d': datetime_pandas})
    res = data_array_validator.validate_coerce(df)
    assert isinstance(res, np.ndarray)
    assert res.dtype == 'object'
    np.testing.assert_array_equal(res, dates_array.reshape(len(dates_array), 1))