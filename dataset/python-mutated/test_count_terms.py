from __future__ import annotations
import pytest
from ansible.module_utils.common.validation import count_terms

@pytest.fixture
def params():
    if False:
        while True:
            i = 10
    return {'name': 'bob', 'dest': '/etc/hosts', 'state': 'present', 'value': 5}

def test_count_terms(params):
    if False:
        i = 10
        return i + 15
    check = set(('name', 'dest'))
    assert count_terms(check, params) == 2

def test_count_terms_str_input(params):
    if False:
        return 10
    check = 'name'
    assert count_terms(check, params) == 1

def test_count_terms_tuple_input(params):
    if False:
        i = 10
        return i + 15
    check = ('name', 'dest')
    assert count_terms(check, params) == 2

def test_count_terms_list_input(params):
    if False:
        i = 10
        return i + 15
    check = ['name', 'dest']
    assert count_terms(check, params) == 2