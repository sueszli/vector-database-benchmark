import pytest
from _plotly_utils.basevalidators import BaseDataValidator
from plotly.graph_objs import Scatter, Bar, Box

@pytest.fixture()
def validator():
    if False:
        for i in range(10):
            print('nop')
    return BaseDataValidator(class_strs_map={'scatter': 'Scatter', 'bar': 'Bar', 'box': 'Box'}, plotly_name='prop', parent_name='parent', set_uid=True)

@pytest.fixture()
def validator_nouid():
    if False:
        print('Hello World!')
    return BaseDataValidator(class_strs_map={'scatter': 'Scatter', 'bar': 'Bar', 'box': 'Box'}, plotly_name='prop', parent_name='parent', set_uid=False)

def test_acceptance(validator):
    if False:
        i = 10
        return i + 15
    val = [Scatter(mode='lines'), Box(fillcolor='yellow')]
    res = validator.validate_coerce(val)
    res_present = validator.present(res)
    assert isinstance(res, list)
    assert isinstance(res_present, tuple)
    assert isinstance(res_present[0], Scatter)
    assert res_present[0].type == 'scatter'
    assert res_present[0].mode == 'lines'
    assert isinstance(res_present[1], Box)
    assert res_present[1].type == 'box'
    assert res_present[1].fillcolor == 'yellow'
    assert res_present[0].uid != res_present[1].uid

def test_acceptance_dict(validator):
    if False:
        print('Hello World!')
    val = (dict(type='scatter', mode='lines'), dict(type='box', fillcolor='yellow'))
    res = validator.validate_coerce(val)
    res_present = validator.present(res)
    assert isinstance(res, list)
    assert isinstance(res_present, tuple)
    assert isinstance(res_present[0], Scatter)
    assert res_present[0].type == 'scatter'
    assert res_present[0].mode == 'lines'
    assert isinstance(res_present[1], Box)
    assert res_present[1].type == 'box'
    assert res_present[1].fillcolor == 'yellow'
    assert res_present[0].uid != res_present[1].uid

def test_default_is_scatter(validator):
    if False:
        i = 10
        return i + 15
    val = [dict(mode='lines')]
    res = validator.validate_coerce(val)
    res_present = validator.present(res)
    assert isinstance(res, list)
    assert isinstance(res_present, tuple)
    assert isinstance(res_present[0], Scatter)
    assert res_present[0].type == 'scatter'
    assert res_present[0].mode == 'lines'

def test_rejection_type(validator):
    if False:
        return 10
    val = 37
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

def test_rejection_element_type(validator):
    if False:
        i = 10
        return i + 15
    val = [42]
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

def test_rejection_element_attr(validator):
    if False:
        i = 10
        return i + 15
    val = [dict(type='scatter', bogus=99)]
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid property specified for object of type ' + "plotly.graph_objs.Scatter: 'bogus'" in str(validation_failure.value)

def test_rejection_element_tracetype(validator):
    if False:
        while True:
            i = 10
    val = [dict(type='bogus', a=4)]
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid element(s)' in str(validation_failure.value)

def test_skip_invalid(validator_nouid):
    if False:
        return 10
    val = (dict(type='scatter', mode='lines', marker={'color': 'green', 'bogus': 23}, line='bad_value'), dict(type='box', fillcolor='yellow', bogus=111), dict(type='bogus', mode='lines+markers', x=[2, 1, 3]))
    expected = [dict(type='scatter', mode='lines', marker={'color': 'green'}), dict(type='box', fillcolor='yellow'), dict(type='scatter', mode='lines+markers', x=[2, 1, 3])]
    res = validator_nouid.validate_coerce(val, skip_invalid=True)
    assert [el.to_plotly_json() for el in res] == expected