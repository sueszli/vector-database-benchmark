from __future__ import annotations
import pytest
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.validation import check_missing_parameters

def test_check_missing_parameters():
    if False:
        while True:
            i = 10
    assert check_missing_parameters([], {}) == []

def test_check_missing_parameters_list():
    if False:
        for i in range(10):
            print('nop')
    expected = 'missing required arguments: path'
    with pytest.raises(TypeError) as e:
        check_missing_parameters({}, ['path'])
    assert to_native(e.value) == expected

def test_check_missing_parameters_positive():
    if False:
        i = 10
        return i + 15
    assert check_missing_parameters({'path': '/foo'}, ['path']) == []