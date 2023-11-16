from __future__ import annotations
import pytest
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.validation import check_type_str
TEST_CASES = (('string', 'string'), (100, '100'), (1.5, '1.5'), ({'k1': 'v1'}, "{'k1': 'v1'}"), ([1, 2, 'three'], "[1, 2, 'three']"), ((1, 2), '(1, 2)'))

@pytest.mark.parametrize('value, expected', TEST_CASES)
def test_check_type_str(value, expected):
    if False:
        return 10
    assert expected == check_type_str(value)

@pytest.mark.parametrize('value, expected', TEST_CASES[1:])
def test_check_type_str_no_conversion(value, expected):
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError) as e:
        check_type_str(value, allow_conversion=False)
    assert 'is not a string and conversion is not allowed' in to_native(e.value)