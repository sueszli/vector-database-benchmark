import pytest
from _plotly_utils.basevalidators import LiteralValidator
import numpy as np

@pytest.fixture()
def validator():
    if False:
        i = 10
        return i + 15
    return LiteralValidator('prop', 'parent', 'scatter')

@pytest.mark.parametrize('val', ['scatter'])
def test_acceptance(val, validator):
    if False:
        while True:
            i = 10
    assert validator.validate_coerce(val) is val

@pytest.mark.parametrize('val', ['hello', (), [], [1, 2, 3], set(), '34'])
def test_rejection(val, validator):
    if False:
        return 10
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'read-only' in str(validation_failure.value)