from __future__ import annotations
import pytest
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.validation import check_type_int

def test_check_type_int():
    if False:
        return 10
    test_cases = (('1', 1), (u'1', 1), (1002, 1002))
    for case in test_cases:
        assert case[1] == check_type_int(case[0])

def test_check_type_int_fail():
    if False:
        while True:
            i = 10
    test_cases = ({'k1': 'v1'}, (b'1', 1), (3.14159, 3), 'b')
    for case in test_cases:
        with pytest.raises(TypeError) as e:
            check_type_int(case)
        assert 'cannot be converted to an int' in to_native(e.value)