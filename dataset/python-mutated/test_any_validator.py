import pytest
from _plotly_utils.basevalidators import AnyValidator
import numpy as np

@pytest.fixture()
def validator():
    if False:
        return 10
    return AnyValidator('prop', 'parent')

@pytest.fixture()
def validator_aok():
    if False:
        for i in range(10):
            print('nop')
    return AnyValidator('prop', 'parent', array_ok=True)

@pytest.mark.parametrize('val', [set(), 'Hello', 123, np.inf, np.nan, {}])
def test_acceptance(val, validator):
    if False:
        i = 10
        return i + 15
    assert validator.validate_coerce(val) is val

@pytest.mark.parametrize('val', [23, 'Hello!', [], (), np.array([]), ('Hello', 'World'), ['Hello', 'World'], [np.pi, np.e, {}]])
def test_acceptance_array(val, validator_aok):
    if False:
        for i in range(10):
            print('nop')
    coerce_val = validator_aok.validate_coerce(val)
    if isinstance(val, np.ndarray):
        assert isinstance(coerce_val, np.ndarray)
        assert coerce_val.dtype == 'object'
        assert np.array_equal(coerce_val, val)
    elif isinstance(val, (list, tuple)):
        assert coerce_val == list(val)
        assert validator_aok.present(coerce_val) == tuple(val)
    else:
        assert coerce_val == val
        assert validator_aok.present(coerce_val) == val