import pytest
from superset.utils.core import form_data_to_adhoc, simple_filter_to_adhoc
from tests.integration_tests.test_app import app

def test_simple_filter_to_adhoc_generates_deterministic_values():
    if False:
        while True:
            i = 10
    input_1 = {'op': 'IS NOT NULL', 'col': 'LATITUDE', 'val': ''}
    input_2 = {**input_1, 'col': 'LONGITUDE'}
    assert simple_filter_to_adhoc(input_1) == simple_filter_to_adhoc(input_1)
    assert simple_filter_to_adhoc(input_1) == {'clause': 'WHERE', 'expressionType': 'SIMPLE', 'comparator': '', 'operator': 'IS NOT NULL', 'subject': 'LATITUDE', 'filterOptionName': '6ac89d498115da22396f80a765cffc70'}
    assert simple_filter_to_adhoc(input_1) != simple_filter_to_adhoc(input_2)
    assert simple_filter_to_adhoc(input_2) == {'clause': 'WHERE', 'expressionType': 'SIMPLE', 'comparator': '', 'operator': 'IS NOT NULL', 'subject': 'LONGITUDE', 'filterOptionName': '9c984bd3714883ca859948354ce26ab9'}

def test_form_data_to_adhoc_generates_deterministic_values():
    if False:
        while True:
            i = 10
    form_data = {'where': '1 = 1', 'having': 'count(*) > 1'}
    assert form_data_to_adhoc(form_data, 'where') == form_data_to_adhoc(form_data, 'where')
    assert form_data_to_adhoc(form_data, 'where') == {'clause': 'WHERE', 'expressionType': 'SQL', 'sqlExpression': '1 = 1', 'filterOptionName': '99fe79985afbddea4492626dc6a87b74'}
    assert form_data_to_adhoc(form_data, 'having') == form_data_to_adhoc(form_data, 'having')
    assert form_data_to_adhoc(form_data, 'having') == {'clause': 'HAVING', 'expressionType': 'SQL', 'sqlExpression': 'count(*) > 1', 'filterOptionName': '1da11f6b709c3190daeabb84f77fc8c2'}

def test_form_data_to_adhoc_incorrect_clause_type():
    if False:
        i = 10
        return i + 15
    form_data = {'where': '1 = 1', 'having': 'count(*) > 1'}
    with pytest.raises(ValueError):
        with app.app_context():
            form_data_to_adhoc(form_data, 'foobar')