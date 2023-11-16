"""
mac_power tests
"""
import pytest
import salt.modules.mac_power as mac_power
from salt.exceptions import SaltInvocationError

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {mac_power: {}}

def test_validate_sleep_valid_number():
    if False:
        while True:
            i = 10
    '\n    test _validate_sleep function with valid number\n    '
    assert mac_power._validate_sleep(179) == 179

def test_validate_sleep_invalid_number():
    if False:
        while True:
            i = 10
    '\n    test _validate_sleep function with invalid number\n    '
    pytest.raises(SaltInvocationError, mac_power._validate_sleep, 181)

def test_validate_sleep_valid_string():
    if False:
        i = 10
        return i + 15
    '\n    test _validate_sleep function with valid string\n    '
    assert mac_power._validate_sleep('never') == 'Never'
    assert mac_power._validate_sleep('off') == 'Never'

def test_validate_sleep_invalid_string():
    if False:
        return 10
    '\n    test _validate_sleep function with invalid string\n    '
    pytest.raises(SaltInvocationError, mac_power._validate_sleep, 'bob')

def test_validate_sleep_bool_true():
    if False:
        while True:
            i = 10
    '\n    test _validate_sleep function with True\n    '
    pytest.raises(SaltInvocationError, mac_power._validate_sleep, True)

def test_validate_sleep_bool_false():
    if False:
        for i in range(10):
            print('nop')
    '\n    test _validate_sleep function with False\n    '
    assert mac_power._validate_sleep(False) == 'Never'

def test_validate_sleep_unexpected():
    if False:
        i = 10
        return i + 15
    '\n    test _validate_sleep function with True\n    '
    pytest.raises(SaltInvocationError, mac_power._validate_sleep, 172.7)