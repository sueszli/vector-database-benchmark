import itertools
import pytest
from _plotly_utils.basevalidators import FlaglistValidator
import numpy as np
EXTRAS = ['none', 'all', True, False, 3]
FLAGS = ['lines', 'markers', 'text']

@pytest.fixture(params=[None, EXTRAS])
def validator(request):
    if False:
        for i in range(10):
            print('nop')
    return FlaglistValidator('prop', 'parent', flags=FLAGS, extras=request.param)

@pytest.fixture()
def validator_extra():
    if False:
        print('Hello World!')
    return FlaglistValidator('prop', 'parent', flags=FLAGS, extras=EXTRAS)

@pytest.fixture()
def validator_extra_aok():
    if False:
        while True:
            i = 10
    return FlaglistValidator('prop', 'parent', flags=FLAGS, extras=EXTRAS, array_ok=True)

@pytest.fixture(params=['+'.join(p) for i in range(1, len(FLAGS) + 1) for p in itertools.permutations(FLAGS, i)])
def flaglist(request):
    if False:
        return 10
    return request.param

@pytest.fixture(params=EXTRAS)
def extra(request):
    if False:
        while True:
            i = 10
    return request.param

def test_acceptance(flaglist, validator):
    if False:
        for i in range(10):
            print('nop')
    assert validator.validate_coerce(flaglist) == flaglist

@pytest.mark.parametrize('in_val,coerce_val', [('  lines ', 'lines'), (' lines + markers ', 'lines+markers'), ('lines ,markers', 'lines+markers')])
def test_coercion(in_val, coerce_val, validator):
    if False:
        while True:
            i = 10
    assert validator.validate_coerce(in_val) == coerce_val

@pytest.mark.parametrize('val', [(), ['lines'], set(), {}])
def test_rejection_type(val, validator):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['', 'line', 'markers+line', 'lin es', 'lin es+markers', 21])
def test_rejection_val(val, validator):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

def test_acceptance_extra(extra, validator_extra):
    if False:
        print('Hello World!')
    assert validator_extra.validate_coerce(extra) == extra

@pytest.mark.parametrize('in_val,coerce_val', [('  none ', 'none'), ('all  ', 'all')])
def test_coercion(in_val, coerce_val, validator_extra):
    if False:
        while True:
            i = 10
    assert validator_extra.validate_coerce(in_val) == coerce_val

@pytest.mark.parametrize('val', ['al l', 'lines+all', 'none+markers', 'markers+lines+text+none'])
def test_rejection_val(val, validator_extra):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as validation_failure:
        validator_extra.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

def test_acceptance_aok_scalar_flaglist(flaglist, validator_extra_aok):
    if False:
        for i in range(10):
            print('nop')
    assert validator_extra_aok.validate_coerce(flaglist) == flaglist

def test_acceptance_aok_scalar_extra(extra, validator_extra_aok):
    if False:
        for i in range(10):
            print('nop')
    assert validator_extra_aok.validate_coerce(extra) == extra

def test_acceptance_aok_scalarlist_flaglist(flaglist, validator_extra_aok):
    if False:
        return 10
    assert np.array_equal(validator_extra_aok.validate_coerce([flaglist]), np.array([flaglist], dtype='unicode'))

@pytest.mark.parametrize('val', [['all', 'markers', 'text+markers'], ['lines', 'lines+markers', 'markers+lines+text'], ['all', 'all', 'lines+text'] + EXTRAS])
def test_acceptance_aok_list_flaglist(val, validator_extra_aok):
    if False:
        i = 10
        return i + 15
    assert np.array_equal(validator_extra_aok.validate_coerce(val), np.array(val, dtype='unicode'))

@pytest.mark.parametrize('in_val,expected', [(['  lines ', ' lines + markers ', 'lines ,markers', '  all  '], ['lines', 'lines+markers', 'lines+markers', 'all']), (np.array(['text   +lines']), np.array(['text+lines'], dtype='unicode'))])
def test_coercion_aok(in_val, expected, validator_extra_aok):
    if False:
        while True:
            i = 10
    coerce_val = validator_extra_aok.validate_coerce(in_val)
    if isinstance(in_val, (list, tuple)):
        expected == coerce_val
        validator_extra_aok.present(coerce_val) == tuple(expected)
    else:
        assert np.array_equal(coerce_val, coerce_val)
        assert np.array_equal(validator_extra_aok.present(coerce_val), coerce_val)

@pytest.mark.parametrize('val', [21, set(), {}])
def test_rejection_aok_type(val, validator_extra_aok):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as validation_failure:
        validator_extra_aok.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', [[21, 'markers'], ['lines', ()], ['none', set()], ['lines+text', {}, 'markers']])
def test_rejection_aok_element_type(val, validator_extra_aok):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as validation_failure:
        validator_extra_aok.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

@pytest.mark.parametrize('val', [['all+markers', 'text+markers'], ['line', 'lines+markers', 'markers+lines+text'], ['all', '', 'lines+text', 'none']])
def test_rejection_aok_element_val(val, validator_extra_aok):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator_extra_aok.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)