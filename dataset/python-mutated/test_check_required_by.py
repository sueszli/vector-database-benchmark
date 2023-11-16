from __future__ import annotations
import pytest
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.validation import check_required_by

@pytest.fixture
def path_arguments_terms():
    if False:
        print('Hello World!')
    return {'path': ['mode', 'owner']}

def test_check_required_by():
    if False:
        return 10
    arguments_terms = {}
    params = {}
    assert check_required_by(arguments_terms, params) == {}

def test_check_required_by_missing():
    if False:
        print('Hello World!')
    arguments_terms = {'force': 'force_reason'}
    params = {'force': True}
    expected = "missing parameter(s) required by 'force': force_reason"
    with pytest.raises(TypeError) as e:
        check_required_by(arguments_terms, params)
    assert to_native(e.value) == expected

def test_check_required_by_multiple(path_arguments_terms):
    if False:
        i = 10
        return i + 15
    params = {'path': '/foo/bar'}
    expected = "missing parameter(s) required by 'path': mode, owner"
    with pytest.raises(TypeError) as e:
        check_required_by(path_arguments_terms, params)
    assert to_native(e.value) == expected

def test_check_required_by_single(path_arguments_terms):
    if False:
        i = 10
        return i + 15
    params = {'path': '/foo/bar', 'mode': '0700'}
    expected = "missing parameter(s) required by 'path': owner"
    with pytest.raises(TypeError) as e:
        check_required_by(path_arguments_terms, params)
    assert to_native(e.value) == expected

def test_check_required_by_missing_none(path_arguments_terms):
    if False:
        i = 10
        return i + 15
    params = {'path': '/foo/bar', 'mode': '0700', 'owner': 'root'}
    assert check_required_by(path_arguments_terms, params)

def test_check_required_by_options_context(path_arguments_terms):
    if False:
        for i in range(10):
            print('nop')
    params = {'path': '/foo/bar', 'mode': '0700'}
    options_context = ['foo_context']
    expected = "missing parameter(s) required by 'path': owner found in foo_context"
    with pytest.raises(TypeError) as e:
        check_required_by(path_arguments_terms, params, options_context)
    assert to_native(e.value) == expected

def test_check_required_by_missing_multiple_options_context(path_arguments_terms):
    if False:
        while True:
            i = 10
    params = {'path': '/foo/bar'}
    options_context = ['foo_context']
    expected = "missing parameter(s) required by 'path': mode, owner found in foo_context"
    with pytest.raises(TypeError) as e:
        check_required_by(path_arguments_terms, params, options_context)
    assert to_native(e.value) == expected