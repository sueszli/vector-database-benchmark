import pytest
from _plotly_utils.basevalidators import SubplotidValidator
import numpy as np

@pytest.fixture()
def validator():
    if False:
        for i in range(10):
            print('nop')
    return SubplotidValidator('prop', 'parent', dflt='geo')

@pytest.mark.parametrize('val', ['geo'] + ['geo%d' % i for i in range(2, 10)])
def test_acceptance(val, validator):
    if False:
        print('Hello World!')
    assert validator.validate_coerce(val) == val

@pytest.mark.parametrize('val', [23, [], {}, set(), np.inf, np.nan])
def test_rejection_type(val, validator):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['', 'bogus', 'geo0'])
def test_rejection_value(val, validator):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)