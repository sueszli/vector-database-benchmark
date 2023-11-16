from __future__ import annotations
import pytest
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.validation import check_type_bits

def test_check_type_bits():
    if False:
        for i in range(10):
            print('nop')
    test_cases = (('1', 1), (99, 99), (1.5, 2), ('1.5', 2), ('2b', 2), ('2k', 2048), ('2K', 2048), ('1m', 1048576), ('1M', 1048576), ('1g', 1073741824), ('1G', 1073741824), (1073741824, 1073741824))
    for case in test_cases:
        assert case[1] == check_type_bits(case[0])

def test_check_type_bits_fail():
    if False:
        while True:
            i = 10
    test_cases = ('foo', '2KB', '1MB', '1GB')
    for case in test_cases:
        with pytest.raises(TypeError) as e:
            check_type_bits(case)
        assert 'cannot be converted to a Bit value' in to_native(e.value)