import pytest
from _plotly_utils.basevalidators import ColorValidator
import numpy as np

@pytest.fixture()
def validator():
    if False:
        for i in range(10):
            print('nop')
    return ColorValidator('prop', 'parent')

@pytest.fixture()
def validator_colorscale():
    if False:
        while True:
            i = 10
    return ColorValidator('prop', 'parent', colorscale_path='parent.colorscale')

@pytest.fixture()
def validator_aok():
    if False:
        while True:
            i = 10
    return ColorValidator('prop', 'parent', array_ok=True)

@pytest.fixture()
def validator_aok_colorscale():
    if False:
        for i in range(10):
            print('nop')
    return ColorValidator('prop', 'parent', array_ok=True, colorscale_path='parent.colorscale')

@pytest.mark.parametrize('val', ['red', 'BLUE', 'rgb(255, 0, 0)', 'var(--accent)', 'hsl(0, 100%, 50%)', 'hsla(0, 100%, 50%, 100%)', 'hsv(0, 100%, 100%)', 'hsva(0, 100%, 100%, 50%)'])
def test_acceptance(val, validator):
    if False:
        while True:
            i = 10
    assert validator.validate_coerce(val) == val

@pytest.mark.parametrize('val', [set(), 23, 0.5, {}, ['red'], [12]])
def test_rejection(val, validator):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['redd', 'rgbbb(255, 0, 0)', 'hsl(0, 1%0000%, 50%)'])
def test_rejection(val, validator):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['red', 'BLUE', 23, 15, 'rgb(255, 0, 0)', 'var(--accent)', 'hsl(0, 100%, 50%)', 'hsla(0, 100%, 50%, 100%)', 'hsv(0, 100%, 100%)', 'hsva(0, 100%, 100%, 50%)'])
def test_acceptance_colorscale(val, validator_colorscale):
    if False:
        return 10
    assert validator_colorscale.validate_coerce(val) == val

@pytest.mark.parametrize('val', [set(), {}, ['red'], [12]])
def test_rejection_colorscale(val, validator_colorscale):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as validation_failure:
        validator_colorscale.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['redd', 'rgbbb(255, 0, 0)', 'hsl(0, 1%0000%, 50%)'])
def test_rejection_colorscale(val, validator_colorscale):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator_colorscale.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['blue', ['red', 'rgb(255, 0, 0)'], np.array(['red', 'rgb(255, 0, 0)']), ['hsl(0, 100%, 50%)', 'hsla(0, 100%, 50%, 100%)', 'hsv(0, 100%, 100%)'], np.array(['hsl(0, 100%, 50%)', 'hsla(0, 100%, 50%, 100%)', 'hsv(0, 100%, 100%)']), ['hsva(0, 100%, 100%, 50%)']])
def test_acceptance_aok(val, validator_aok):
    if False:
        return 10
    coerce_val = validator_aok.validate_coerce(val)
    if isinstance(val, np.ndarray):
        assert np.array_equal(coerce_val, val)
    elif isinstance(val, list):
        assert validator_aok.present(coerce_val) == tuple(val)
    else:
        assert coerce_val == val

@pytest.mark.parametrize('val', ['green', [['blue']], [['red', 'rgb(255, 0, 0)'], ['hsl(0, 100%, 50%)', 'hsla(0, 100%, 50%, 100%)']], np.array([['red', 'rgb(255, 0, 0)'], ['hsl(0, 100%, 50%)', 'hsla(0, 100%, 50%, 100%)']])])
def test_acceptance_aok_2D(val, validator_aok):
    if False:
        while True:
            i = 10
    coerce_val = validator_aok.validate_coerce(val)
    if isinstance(val, np.ndarray):
        assert np.array_equal(coerce_val, val)
    elif isinstance(val, list):
        assert validator_aok.present(coerce_val) == tuple(val)
    else:
        assert coerce_val == val

@pytest.mark.parametrize('val', [[23], [0, 1, 2], ['redd', 'rgb(255, 0, 0)'], ['hsl(0, 100%, 50_00%)', 'hsla(0, 100%, 50%, 100%)', 'hsv(0, 100%, 100%)'], ['hsva(0, 1%00%, 100%, 50%)']])
def test_rejection_aok(val, validator_aok):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as validation_failure:
        validator_aok.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

@pytest.mark.parametrize('val', [[['redd', 'rgb(255, 0, 0)']], [['hsl(0, 100%, 50_00%)', 'hsla(0, 100%, 50%, 100%)'], ['hsv(0, 100%, 100%)', 'purple']], [np.array(['hsl(0, 100%, 50_00%)', 'hsla(0, 100%, 50%, 100%)']), np.array(['hsv(0, 100%, 100%)', 'purple'])], [['blue'], [2]]])
def test_rejection_aok_2D(val, validator_aok):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError) as validation_failure:
        validator_aok.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['blue', 23, [0, 1, 2], ['red', 0.5, 'rgb(255, 0, 0)'], ['hsl(0, 100%, 50%)', 'hsla(0, 100%, 50%, 100%)', 'hsv(0, 100%, 100%)'], ['hsva(0, 100%, 100%, 50%)']])
def test_acceptance_aok_colorscale(val, validator_aok_colorscale):
    if False:
        print('Hello World!')
    coerce_val = validator_aok_colorscale.validate_coerce(val)
    if isinstance(val, (list, np.ndarray)):
        assert np.array_equal(list(coerce_val), val)
    else:
        assert coerce_val == val

@pytest.mark.parametrize('val', [['redd', 0.5, 'rgb(255, 0, 0)'], ['hsl(0, 100%, 50_00%)', 'hsla(0, 100%, 50%, 100%)', 'hsv(0, 100%, 100%)'], ['hsva(0, 1%00%, 100%, 50%)']])
def test_rejection_aok_colorscale(val, validator_aok_colorscale):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError) as validation_failure:
        validator_aok_colorscale.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

def test_description(validator):
    if False:
        i = 10
        return i + 15
    desc = validator.description()
    assert 'A number that will be interpreted as a color' not in desc
    assert 'A list or array of any of the above' not in desc

def test_description_aok(validator_aok):
    if False:
        while True:
            i = 10
    desc = validator_aok.description()
    assert 'A number that will be interpreted as a color' not in desc
    assert 'A list or array of any of the above' in desc

def test_description_aok_colorscale(validator_aok_colorscale):
    if False:
        i = 10
        return i + 15
    desc = validator_aok_colorscale.description()
    assert 'A number that will be interpreted as a color' in desc
    assert 'A list or array of any of the above' in desc