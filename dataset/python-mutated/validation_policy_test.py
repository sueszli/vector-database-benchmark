from datetime import datetime, timedelta
import pytest
from scylla.database import ProxyIP
from scylla.validation_policy import ValidationPolicy

@pytest.fixture
def p():
    if False:
        print('Hello World!')
    return ProxyIP(ip='127.0.0.1', port=3306, is_valid=False)

@pytest.fixture
def valid_http_proxy():
    if False:
        print('Hello World!')
    return ProxyIP(ip='127.0.0.1', port=3306, is_valid=True)

def test_should_validate_policy_attempts_0(p: ProxyIP):
    if False:
        for i in range(10):
            print('nop')
    policy = ValidationPolicy(proxy_ip=p)
    assert policy.should_validate()

def test_should_validate_policy_attempts_1(p: ProxyIP):
    if False:
        while True:
            i = 10
    p.attempts = 1
    policy = ValidationPolicy(proxy_ip=p)
    assert policy.should_validate()

def test_should_validate_policy_attempts_3(p: ProxyIP):
    if False:
        return 10
    p.attempts = 3
    policy = ValidationPolicy(proxy_ip=p)
    assert not policy.should_validate()

def test_should_validate_policy_attempts_3_after_24h_in_48h(p: ProxyIP):
    if False:
        i = 10
        return i + 15
    p.attempts = 3
    p.created_at = datetime.now() - timedelta(hours=25)
    policy = ValidationPolicy(proxy_ip=p)
    assert policy.should_validate()

def test_should_try_https(valid_http_proxy: ProxyIP):
    if False:
        i = 10
        return i + 15
    valid_http_proxy.attempts = 1
    policy = ValidationPolicy(proxy_ip=valid_http_proxy)
    assert policy.should_try_https()

def test_should_try_https_attempts_3(valid_http_proxy: ProxyIP):
    if False:
        while True:
            i = 10
    valid_http_proxy.attempts = 3
    policy = ValidationPolicy(proxy_ip=valid_http_proxy)
    assert not policy.should_try_https()