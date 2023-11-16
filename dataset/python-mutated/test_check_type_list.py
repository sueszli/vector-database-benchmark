from __future__ import annotations
import pytest
from ansible.module_utils.common.validation import check_type_list

def test_check_type_list():
    if False:
        for i in range(10):
            print('nop')
    test_cases = (([1, 2], [1, 2]), (1, ['1']), (['a', 'b'], ['a', 'b']), ('a', ['a']), (3.14159, ['3.14159']), ('a,b,1,2', ['a', 'b', '1', '2']))
    for case in test_cases:
        assert case[1] == check_type_list(case[0])

def test_check_type_list_failure():
    if False:
        print('Hello World!')
    test_cases = ({'k1': 'v1'},)
    for case in test_cases:
        with pytest.raises(TypeError):
            check_type_list(case)