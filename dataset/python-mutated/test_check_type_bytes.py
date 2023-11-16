from __future__ import annotations
import pytest
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.validation import check_type_bytes

def test_check_type_bytes():
    if False:
        i = 10
        return i + 15
    test_cases = (('1', 1), (99, 99), (1.5, 2), ('1.5', 2), ('2b', 2), ('2B', 2), ('2k', 2048), ('2K', 2048), ('2KB', 2048), ('1m', 1048576), ('1M', 1048576), ('1MB', 1048576), ('1g', 1073741824), ('1G', 1073741824), ('1GB', 1073741824), (1073741824, 1073741824))
    for case in test_cases:
        assert case[1] == check_type_bytes(case[0])

def test_check_type_bytes_fail():
    if False:
        print('Hello World!')
    test_cases = ('foo', '2kb', '2Kb', '1mb', '1Mb', '1gb', '1Gb')
    for case in test_cases:
        with pytest.raises(TypeError) as e:
            check_type_bytes(case)
        assert 'cannot be converted to a Byte value' in to_native(e.value)