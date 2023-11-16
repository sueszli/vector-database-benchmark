import pytest
from pytest import approx
from _plotly_utils.basevalidators import IntegerValidator
import numpy as np
import pandas as pd

@pytest.fixture()
def validator():
    if False:
        for i in range(10):
            print('nop')
    return IntegerValidator('prop', 'parent')

@pytest.fixture
def validator_min_max():
    if False:
        print('Hello World!')
    return IntegerValidator('prop', 'parent', min=-1, max=2)

@pytest.fixture
def validator_min():
    if False:
        while True:
            i = 10
    return IntegerValidator('prop', 'parent', min=-1)

@pytest.fixture
def validator_max():
    if False:
        print('Hello World!')
    return IntegerValidator('prop', 'parent', max=2)

@pytest.fixture
def validator_aok(request):
    if False:
        print('Hello World!')
    return IntegerValidator('prop', 'parent', min=-2, max=10, array_ok=True)

@pytest.mark.parametrize('val', [1, -19, 0, -1234])
def test_acceptance(val, validator):
    if False:
        i = 10
        return i + 15
    assert validator.validate_coerce(val) == val

@pytest.mark.parametrize('val', ['hello', (), [], [1, 2, 3], set(), '34', np.nan, np.inf, -np.inf])
def test_rejection_by_value(val, validator):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', [0, 1, -1, 2])
def test_acceptance_min_max(val, validator_min_max):
    if False:
        print('Hello World!')
    assert validator_min_max.validate_coerce(val) == approx(val)

@pytest.mark.parametrize('val', [-1.01, -10, 2.1, 3, np.iinfo(int).max, np.iinfo(int).min])
def test_rejection_min_max(val, validator_min_max):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator_min_max.validate_coerce(val)
    assert 'in the interval [-1, 2]' in str(validation_failure.value)

@pytest.mark.parametrize('val', [-1, 0, 1, 23, 99999])
def test_acceptance_min(val, validator_min):
    if False:
        return 10
    assert validator_min.validate_coerce(val) == approx(val)

@pytest.mark.parametrize('val', [-2, -123, np.iinfo(int).min])
def test_rejection_min(val, validator_min):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as validation_failure:
        validator_min.validate_coerce(val)
    assert 'in the interval [-1, 9223372036854775807]' in str(validation_failure.value)

@pytest.mark.parametrize('val', [1, 2, -10, -999999, np.iinfo(np.int32).min])
def test_acceptance_max(val, validator_max):
    if False:
        print('Hello World!')
    assert validator_max.validate_coerce(val) == approx(val)

@pytest.mark.parametrize('val', [3, 10, np.iinfo(np.int32).max])
def test_rejection_max(val, validator_max):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as validation_failure:
        validator_max.validate_coerce(val)
    assert 'in the interval [-9223372036854775808, 2]' in str(validation_failure.value)

@pytest.mark.parametrize('val', [-2, 1, 0, 1, 10])
def test_acceptance_aok_scalars(val, validator_aok):
    if False:
        while True:
            i = 10
    assert validator_aok.validate_coerce(val) == val

@pytest.mark.parametrize('val', [[1, 0], [1], [-2, 1, 8], np.array([3, 2, -1, 5])])
def test_acceptance_aok_list(val, validator_aok):
    if False:
        while True:
            i = 10
    assert np.array_equal(validator_aok.validate_coerce(val), val)

@pytest.mark.parametrize('val,expected', [([1, 0], (1, 0)), ((1, -1), (1, -1)), (np.array([-1, 0, 5.0], dtype='int16'), [-1, 0, 5]), (np.array([1, 0], dtype=np.int64), [1, 0]), (pd.Series([1, 0], dtype=np.int64), [1, 0]), (pd.Index([1, 0], dtype=np.int64), [1, 0])])
def test_coercion_aok_list(val, expected, validator_aok):
    if False:
        i = 10
        return i + 15
    v = validator_aok.validate_coerce(val)
    if isinstance(val, (np.ndarray, pd.Series, pd.Index)):
        assert v.dtype == val.dtype
        assert np.array_equal(validator_aok.present(v), np.array(expected, dtype=np.int32))
    else:
        assert isinstance(v, list)
        assert validator_aok.present(v) == expected

@pytest.mark.parametrize('val', [['a', 4], [[], 3, 4]])
def test_integer_validator_rejection_aok(val, validator_aok):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator_aok.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

@pytest.mark.parametrize('val', [[-1, 11], [1.5, -3], [0, np.iinfo(np.int32).max], [0, np.iinfo(np.int32).min]])
def test_rejection_aok_min_max(val, validator_aok):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator_aok.validate_coerce(val)
    assert 'in the interval [-2, 10]' in str(validation_failure.value)