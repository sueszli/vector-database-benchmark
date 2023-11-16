from __future__ import annotations
import pytest
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.validation import check_required_arguments

@pytest.fixture
def arguments_terms():
    if False:
        return 10
    return {'foo': {'required': True}, 'bar': {'required': False}, 'tomato': {'irrelevant': 72}}

@pytest.fixture
def arguments_terms_multiple():
    if False:
        while True:
            i = 10
    return {'foo': {'required': True}, 'bar': {'required': True}, 'tomato': {'irrelevant': 72}}

def test_check_required_arguments(arguments_terms):
    if False:
        return 10
    params = {'foo': 'hello', 'bar': 'haha'}
    assert check_required_arguments(arguments_terms, params) == []

def test_check_required_arguments_missing(arguments_terms):
    if False:
        print('Hello World!')
    params = {'apples': 'woohoo'}
    expected = 'missing required arguments: foo'
    with pytest.raises(TypeError) as e:
        check_required_arguments(arguments_terms, params)
    assert to_native(e.value) == expected

def test_check_required_arguments_missing_multiple(arguments_terms_multiple):
    if False:
        while True:
            i = 10
    params = {'apples': 'woohoo'}
    expected = 'missing required arguments: bar, foo'
    with pytest.raises(TypeError) as e:
        check_required_arguments(arguments_terms_multiple, params)
    assert to_native(e.value) == expected

def test_check_required_arguments_missing_none():
    if False:
        for i in range(10):
            print('nop')
    terms = None
    params = {'foo': 'bar', 'baz': 'buzz'}
    assert check_required_arguments(terms, params) == []

def test_check_required_arguments_no_params(arguments_terms):
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError) as te:
        check_required_arguments(arguments_terms, None)
    assert "'NoneType' is not iterable" in to_native(te.value)