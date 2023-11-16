import pytest
from _plotly_utils.basevalidators import StringValidator
import numpy as np

@pytest.fixture()
def validator():
    if False:
        return 10
    return StringValidator('prop', 'parent')

@pytest.fixture()
def validator_values():
    if False:
        print('Hello World!')
    return StringValidator('prop', 'parent', values=['foo', 'BAR', ''])

@pytest.fixture()
def validator_no_blanks():
    if False:
        return 10
    return StringValidator('prop', 'parent', no_blank=True)

@pytest.fixture()
def validator_strict():
    if False:
        return 10
    return StringValidator('prop', 'parent', strict=True)

@pytest.fixture
def validator_aok():
    if False:
        i = 10
        return i + 15
    return StringValidator('prop', 'parent', array_ok=True, strict=False)

@pytest.fixture
def validator_aok_strict():
    if False:
        for i in range(10):
            print('nop')
    return StringValidator('prop', 'parent', array_ok=True, strict=True)

@pytest.fixture
def validator_aok_values():
    if False:
        for i in range(10):
            print('nop')
    return StringValidator('prop', 'parent', values=['foo', 'BAR', '', 'baz'], array_ok=True)

@pytest.fixture()
def validator_no_blanks_aok():
    if False:
        print('Hello World!')
    return StringValidator('prop', 'parent', no_blank=True, array_ok=True)

@pytest.mark.parametrize('val', ['bar', 234, np.nan, 'HELLO!!!', 'world!@#$%^&*()', '', 'μ'])
def test_acceptance(val, validator):
    if False:
        print('Hello World!')
    expected = str(val) if not isinstance(val, str) else val
    assert validator.validate_coerce(val) == expected

@pytest.mark.parametrize('val', [(), [], [1, 2, 3], set()])
def test_rejection(val, validator):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['foo', 'BAR', ''])
def test_acceptance_values(val, validator_values):
    if False:
        for i in range(10):
            print('nop')
    assert validator_values.validate_coerce(val) == val

@pytest.mark.parametrize('val', ['FOO', 'bar', 'other', '1234'])
def test_rejection_values(val, validator_values):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as validation_failure:
        validator_values.validate_coerce(val)
    assert 'Invalid value'.format(val=val) in str(validation_failure.value)
    assert "['foo', 'BAR', '']" in str(validation_failure.value)

@pytest.mark.parametrize('val', ['bar', 'HELLO!!!', 'world!@#$%^&*()', 'μ'])
def test_acceptance_no_blanks(val, validator_no_blanks):
    if False:
        for i in range(10):
            print('nop')
    assert validator_no_blanks.validate_coerce(val) == val

@pytest.mark.parametrize('val', [''])
def test_rejection_no_blanks(val, validator_no_blanks):
    if False:
        return 10
    with pytest.raises(ValueError) as validation_failure:
        validator_no_blanks.validate_coerce(val)
    assert 'A non-empty string' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['bar', 'HELLO!!!', 'world!@#$%^&*()', '', 'μ'])
def test_acceptance_strict(val, validator_strict):
    if False:
        print('Hello World!')
    assert validator_strict.validate_coerce(val) == val

@pytest.mark.parametrize('val', [(), [], [1, 2, 3], set(), np.nan, np.pi, 23])
def test_rejection_strict(val, validator_strict):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as validation_failure:
        validator_strict.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['foo', 'BAR', '', 'baz', 'μ'])
def test_acceptance_aok_scalars(val, validator_aok):
    if False:
        print('Hello World!')
    assert validator_aok.validate_coerce(val) == val

@pytest.mark.parametrize('val', ['foo', ['foo'], np.array(['BAR', '', 'μ'], dtype='object'), ['baz', 'baz', 'baz'], ['foo', None, 'bar', 'μ']])
def test_acceptance_aok_list(val, validator_aok):
    if False:
        while True:
            i = 10
    coerce_val = validator_aok.validate_coerce(val)
    if isinstance(val, np.ndarray):
        assert isinstance(coerce_val, np.ndarray)
        assert np.array_equal(coerce_val, np.array(val, dtype=coerce_val.dtype))
    elif isinstance(val, list):
        assert validator_aok.present(val) == tuple(val)
    else:
        assert coerce_val == val

@pytest.mark.parametrize('val', [['foo', ()], ['foo', 3, 4], [3, 2, 1]])
def test_rejection_aok(val, validator_aok_strict):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator_aok_strict.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

@pytest.mark.parametrize('val', [['foo', 'bar'], ['3', '4'], ['BAR', 'BAR', 'hello!'], ['foo', None]])
def test_rejection_aok_values(val, validator_aok_values):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator_aok_values.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

@pytest.mark.parametrize('val', ['123', ['bar', 'HELLO!!!'], np.array(['bar', 'HELLO!!!'], dtype='object'), ['world!@#$%^&*()', 'μ']])
def test_acceptance_no_blanks_aok(val, validator_no_blanks_aok):
    if False:
        print('Hello World!')
    coerce_val = validator_no_blanks_aok.validate_coerce(val)
    if isinstance(val, np.ndarray):
        assert np.array_equal(coerce_val, np.array(val, dtype=coerce_val.dtype))
    elif isinstance(val, list):
        assert validator_no_blanks_aok.present(coerce_val) == tuple(val)
    else:
        assert coerce_val == val

@pytest.mark.parametrize('val', ['', ['foo', 'bar', ''], np.array(['foo', 'bar', ''], dtype='object'), [''], np.array([''], dtype='object')])
def test_rejection_no_blanks_aok(val, validator_no_blanks_aok):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as validation_failure:
        validator_no_blanks_aok.validate_coerce(val)
    assert 'A non-empty string' in str(validation_failure.value)