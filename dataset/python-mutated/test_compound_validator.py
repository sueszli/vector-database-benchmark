import pytest
from _plotly_utils.basevalidators import CompoundValidator
from plotly.graph_objs.scatter import Marker

@pytest.fixture()
def validator():
    if False:
        for i in range(10):
            print('nop')
    return CompoundValidator('prop', 'scatter', data_class_str='Marker', data_docs='')

def test_acceptance(validator):
    if False:
        return 10
    val = Marker(color='green', size=10)
    res = validator.validate_coerce(val)
    assert isinstance(res, Marker)
    assert res.color == 'green'
    assert res.size == 10

def test_acceptance_none(validator):
    if False:
        print('Hello World!')
    val = None
    res = validator.validate_coerce(val)
    assert isinstance(res, Marker)
    assert res.color is None
    assert res.size is None

def test_acceptance_dict(validator):
    if False:
        while True:
            i = 10
    val = dict(color='green', size=10)
    res = validator.validate_coerce(val)
    assert isinstance(res, Marker)
    assert res.color == 'green'
    assert res.size == 10

def test_rejection_type(validator):
    if False:
        return 10
    val = 37
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

def test_rejection_value(validator):
    if False:
        print('Hello World!')
    val = dict(color='green', size=10, bogus=99)
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert "Invalid property specified for object of type plotly.graph_objs.scatter.Marker: 'bogus'" in str(validation_failure.value)

def test_skip_invalid(validator):
    if False:
        while True:
            i = 10
    val = dict(color='green', size=10, bogus=99, colorbar={'bgcolor': 'blue', 'bogus_inner': 23}, opacity='bogus value')
    expected = dict(color='green', size=10, colorbar={'bgcolor': 'blue'})
    res = validator.validate_coerce(val, skip_invalid=True)
    assert res.to_plotly_json() == expected

def test_skip_invalid_empty_object(validator):
    if False:
        for i in range(10):
            print('nop')
    val = dict(color='green', size=10, colorbar={'bgcolor': 'bad_color', 'bogus_inner': 23}, opacity=0.5)
    expected = dict(color='green', size=10, opacity=0.5)
    res = validator.validate_coerce(val, skip_invalid=True)
    assert res.to_plotly_json() == expected