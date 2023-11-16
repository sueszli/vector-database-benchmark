from __future__ import annotations
import pytest
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.validation import check_required_together

@pytest.fixture
def together_terms():
    if False:
        return 10
    return [['bananas', 'potatoes'], ['cats', 'wolves']]

def test_check_required_together(together_terms):
    if False:
        print('Hello World!')
    params = {'bananas': 'hello', 'potatoes': 'this is here too', 'dogs': 'haha'}
    assert check_required_together(together_terms, params) == []

def test_check_required_together_missing(together_terms):
    if False:
        i = 10
        return i + 15
    params = {'bananas': 'woohoo', 'wolves': 'uh oh'}
    expected = 'parameters are required together: bananas, potatoes'
    with pytest.raises(TypeError) as e:
        check_required_together(together_terms, params)
    assert to_native(e.value) == expected

def test_check_required_together_missing_none():
    if False:
        for i in range(10):
            print('nop')
    terms = None
    params = {'foo': 'bar', 'baz': 'buzz'}
    assert check_required_together(terms, params) == []

def test_check_required_together_no_params(together_terms):
    if False:
        print('Hello World!')
    with pytest.raises(TypeError) as te:
        check_required_together(together_terms, None)
    assert "'NoneType' object is not iterable" in to_native(te.value)