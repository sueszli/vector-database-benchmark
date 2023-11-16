import pytest
from _plotly_utils.basevalidators import CompoundArrayValidator
from plotly.graph_objs.layout import Image

@pytest.fixture()
def validator():
    if False:
        i = 10
        return i + 15
    return CompoundArrayValidator('prop', 'layout', data_class_str='Image', data_docs='')

def test_acceptance(validator):
    if False:
        i = 10
        return i + 15
    val = [Image(opacity=0.5, sizex=120), Image(x=99)]
    res = validator.validate_coerce(val)
    res_present = validator.present(res)
    assert isinstance(res, list)
    assert isinstance(res_present, tuple)
    assert isinstance(res_present[0], Image)
    assert res_present[0].opacity == 0.5
    assert res_present[0].sizex == 120
    assert res_present[0].x is None
    assert isinstance(res_present[1], Image)
    assert res_present[1].opacity is None
    assert res_present[1].sizex is None
    assert res_present[1].x == 99

def test_acceptance_empty(validator):
    if False:
        for i in range(10):
            print('nop')
    val = [{}]
    res = validator.validate_coerce(val)
    res_present = validator.present(res)
    assert isinstance(res, list)
    assert isinstance(res_present, tuple)
    assert isinstance(res_present[0], Image)
    assert res_present[0].opacity is None
    assert res_present[0].sizex is None
    assert res_present[0].x is None

def test_acceptance_dict(validator):
    if False:
        while True:
            i = 10
    val = [dict(opacity=0.5, sizex=120), dict(x=99)]
    res = validator.validate_coerce(val)
    res_present = validator.present(res)
    assert isinstance(res, list)
    assert isinstance(res_present, tuple)
    assert isinstance(res_present[0], Image)
    assert res_present[0].opacity == 0.5
    assert res_present[0].sizex == 120
    assert res_present[0].x is None
    assert isinstance(res[1], Image)
    assert res_present[1].opacity is None
    assert res_present[1].sizex is None
    assert res_present[1].x == 99

def test_rejection_type(validator):
    if False:
        while True:
            i = 10
    val = 37
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)

def test_rejection_element(validator):
    if False:
        print('Hello World!')
    val = ['a', 37]
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

def test_rejection_value(validator):
    if False:
        i = 10
        return i + 15
    val = [dict(opacity=0.5, sizex=120, bogus=100)]
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid property specified for object of type plotly.graph_objs.layout.Image' in str(validation_failure.value)

def test_skip_invalid(validator):
    if False:
        while True:
            i = 10
    val = [dict(opacity='bad_opacity', x=23, sizex=120), dict(x=99, bogus={'a': 23}, sizey=300)]
    expected = [dict(x=23, sizex=120), dict(x=99, sizey=300)]
    res = validator.validate_coerce(val, skip_invalid=True)
    assert [el.to_plotly_json() for el in res] == expected