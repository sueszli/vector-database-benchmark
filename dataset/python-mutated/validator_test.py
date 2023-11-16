import pytest
from scylla.validator import Validator

@pytest.fixture
def validator():
    if False:
        while True:
            i = 10
    return Validator(host='145.239.185.126', port=1080)

@pytest.fixture
def validator2():
    if False:
        return 10
    return Validator(host='162.246.200.100', port=80)

def test_latency(validator):
    if False:
        print('Hello World!')
    validator.validate_latency()
    assert validator.success_rate >= 0
    assert validator.latency >= 0

def test_proxy(validator):
    if False:
        i = 10
        return i + 15
    validator.validate_proxy()

def test_proxy(validator2):
    if False:
        for i in range(10):
            print('nop')
    validator2.validate_proxy()

def test_proxy(validator2, mocker):
    if False:
        print('Hello World!')
    l = mocker.patch('scylla.validator.Validator.validate_latency')
    p = mocker.patch('scylla.validator.Validator.validate_proxy')
    validator2.validate()
    l.assert_called_once()
    p.assert_called_once()