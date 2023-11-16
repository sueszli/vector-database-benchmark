import pytest
import numpy as np
import xarray
import datetime
from _plotly_utils.basevalidators import NumberValidator, IntegerValidator, DataArrayValidator, ColorValidator

@pytest.fixture
def data_array_validator(request):
    if False:
        for i in range(10):
            print('nop')
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
        i = 10
        return i + 15
    return ColorValidator('prop', 'parent', array_ok=True, colorscale_path='')

@pytest.fixture(params=['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64'])
def numeric_dtype(request):
    if False:
        print('Hello World!')
    return request.param

@pytest.fixture(params=[xarray.DataArray])
def xarray_type(request):
    if False:
        print('Hello World!')
    return request.param

@pytest.fixture
def numeric_xarray(request, xarray_type, numeric_dtype):
    if False:
        for i in range(10):
            print('nop')
    return xarray_type(np.arange(10, dtype=numeric_dtype))

@pytest.fixture
def color_object_xarray(request, xarray_type):
    if False:
        i = 10
        return i + 15
    return xarray_type(['blue', 'green', 'red'] * 3)

def test_numeric_validator_numeric_xarray(number_validator, numeric_xarray):
    if False:
        while True:
            i = 10
    res = number_validator.validate_coerce(numeric_xarray)
    assert isinstance(res, np.ndarray)
    assert res.dtype == numeric_xarray.dtype
    np.testing.assert_array_equal(res, numeric_xarray)

def test_integer_validator_numeric_xarray(integer_validator, numeric_xarray):
    if False:
        print('Hello World!')
    res = integer_validator.validate_coerce(numeric_xarray)
    assert isinstance(res, np.ndarray)
    if numeric_xarray.dtype.kind in ('u', 'i'):
        assert res.dtype == numeric_xarray.dtype
    else:
        assert res.dtype == 'int32'
    np.testing.assert_array_equal(res, numeric_xarray)

def test_data_array_validator(data_array_validator, numeric_xarray):
    if False:
        return 10
    res = data_array_validator.validate_coerce(numeric_xarray)
    assert isinstance(res, np.ndarray)
    assert res.dtype == numeric_xarray.dtype
    np.testing.assert_array_equal(res, numeric_xarray)

def test_color_validator_numeric(color_validator, numeric_xarray):
    if False:
        return 10
    res = color_validator.validate_coerce(numeric_xarray)
    assert isinstance(res, np.ndarray)
    assert res.dtype == numeric_xarray.dtype
    np.testing.assert_array_equal(res, numeric_xarray)

def test_color_validator_object(color_validator, color_object_xarray):
    if False:
        print('Hello World!')
    res = color_validator.validate_coerce(color_object_xarray)
    assert isinstance(res, np.ndarray)
    assert res.dtype == 'object'
    np.testing.assert_array_equal(res, color_object_xarray)