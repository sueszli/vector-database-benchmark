import pytest
from _plotly_utils.basevalidators import BooleanValidator
import numpy as np

@pytest.fixture(params=[True, False])
def validator(request):
    if False:
        i = 10
        return i + 15
    return BooleanValidator('prop', 'parent', dflt=request.param)

@pytest.mark.parametrize('val', [True, False])
def test_acceptance(val, validator):
    if False:
        while True:
            i = 10
    assert val == validator.validate_coerce(val)

@pytest.mark.parametrize('val', [1.0, 0.0, 'True', 'False', [], 0, np.nan])
def test_rejection(val, validator):
    if False:
        return 10
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)