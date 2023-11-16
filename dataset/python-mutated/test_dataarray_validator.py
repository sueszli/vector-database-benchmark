import pytest
from _plotly_utils.basevalidators import DataArrayValidator
import numpy as np
import pandas as pd

@pytest.fixture()
def validator():
    if False:
        return 10
    return DataArrayValidator('prop', 'parent')

@pytest.mark.parametrize('val', [[], [1], [''], (), ('Hello, ', 'world!'), ['A', 1, 'B', 0, 'C'], [np.array(1), np.array(2)]])
def test_validator_acceptance_simple(val, validator):
    if False:
        i = 10
        return i + 15
    coerce_val = validator.validate_coerce(val)
    assert isinstance(coerce_val, list)
    assert validator.present(coerce_val) == tuple(val)

@pytest.mark.parametrize('val', [np.array([2, 3, 4]), pd.Series(['a', 'b', 'c']), np.array([[1, 2, 3], [4, 5, 6]])])
def test_validator_acceptance_homogeneous(val, validator):
    if False:
        print('Hello World!')
    coerce_val = validator.validate_coerce(val)
    assert isinstance(coerce_val, np.ndarray)
    assert np.array_equal(validator.present(coerce_val), val)

@pytest.mark.parametrize('val', ['Hello', 23, set(), {}])
def test_rejection(val, validator):
    if False:
        return 10
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)