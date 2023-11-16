from __future__ import annotations
import pytest
from ansible.module_utils.common.validation import check_type_dict

def test_check_type_dict():
    if False:
        print('Hello World!')
    test_cases = (({'k1': 'v1'}, {'k1': 'v1'}), ('k1=v1,k2=v2', {'k1': 'v1', 'k2': 'v2'}), ('k1=v1, k2=v2', {'k1': 'v1', 'k2': 'v2'}), ('k1=v1,     k2=v2,  k3=v3', {'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}), ('{"key": "value", "list": ["one", "two"]}', {'key': 'value', 'list': ['one', 'two']}))
    for case in test_cases:
        assert case[1] == check_type_dict(case[0])

def test_check_type_dict_fail():
    if False:
        return 10
    test_cases = (1, 3.14159, [1, 2], 'a')
    for case in test_cases:
        with pytest.raises(TypeError):
            check_type_dict(case)