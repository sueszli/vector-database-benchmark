import pytest
from _plotly_utils.basevalidators import ColorscaleValidator
from _plotly_utils import colors
import numpy as np
import inspect
import itertools

@pytest.fixture()
def validator():
    if False:
        return 10
    return ColorscaleValidator('prop', 'parent')
colorscale_members = itertools.chain(inspect.getmembers(colors.sequential), inspect.getmembers(colors.diverging), inspect.getmembers(colors.cyclical))
named_colorscales = {c[0]: c[1] for c in colorscale_members if isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], str) and isinstance(c[1], list) and (not c[0].startswith('_'))}

@pytest.fixture(params=list(named_colorscales))
def named_colorscale(request):
    if False:
        i = 10
        return i + 15
    return request.param

@pytest.fixture(params=list(named_colorscales))
def seqence_colorscale(request):
    if False:
        print('Hello World!')
    return named_colorscales[request.param]

def test_acceptance_named(named_colorscale, validator):
    if False:
        return 10
    d = len(named_colorscales[named_colorscale]) - 1
    expected = [[1.0 * i / (1.0 * d), x] for (i, x) in enumerate(named_colorscales[named_colorscale])]
    assert validator.validate_coerce(named_colorscale) == expected
    assert validator.validate_coerce(named_colorscale.upper()) == expected
    expected_tuples = tuple(((c[0], c[1]) for c in expected))
    assert validator.present(expected) == expected_tuples

def test_acceptance_sequence(seqence_colorscale, validator):
    if False:
        return 10
    d = len(seqence_colorscale) - 1
    expected = [[1.0 * i / (1.0 * d), x] for (i, x) in enumerate(seqence_colorscale)]
    assert validator.validate_coerce(seqence_colorscale) == expected
    expected_tuples = tuple(((c[0], c[1]) for c in expected))
    assert validator.present(expected) == expected_tuples

@pytest.mark.parametrize('val', [((0, 'red'),), ((0.1, 'rgb(255,0,0)'), (0.3, 'green')), ((0, 'purple'), (0.2, 'yellow'), (1.0, 'rgba(255,0,0,100)'))])
def test_acceptance_array(val, validator):
    if False:
        for i in range(10):
            print('nop')
    assert validator.validate_coerce(val) == val

@pytest.mark.parametrize('val', [([0, 'red'],), [(0.1, 'rgb(255, 0, 0)'), (0.3, 'GREEN')], (np.array([0, 'Purple'], dtype='object'), (0.2, 'yellow'), (1.0, 'RGBA(255,0,0,100)'))])
def test_acceptance_array(val, validator):
    if False:
        return 10
    expected = [[e[0], e[1]] for e in val]
    coerce_val = validator.validate_coerce(val)
    assert coerce_val == expected
    expected_present = tuple([tuple(e) for e in expected])
    assert validator.present(coerce_val) == expected_present

@pytest.mark.parametrize('val', [23, set(), {}, np.pi])
def test_rejection_type(val, validator):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['Invalid', ''])
def test_rejection_str_value(val, validator):
    if False:
        return 10
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', [[0, 'red'], [[0.1, 'rgb(255,0,0)', None], (0.3, 'green')], ([1.1, 'purple'], [0.2, 'yellow']), ([0.1, 'purple'], [-0.2, 'yellow']), ([0.1, 'purple'], [0.2, 123]), ([0.1, 'purple'], [0.2, 'yellowww'])])
def test_rejection_array(val, validator):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)