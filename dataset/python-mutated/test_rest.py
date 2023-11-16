import pytest
import salt.auth.rest as rest
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    '\n    Rest module configuration\n    '
    return {rest: {'__opts__': {'external_auth': {'rest': {'^url': 'https://test_url/rest', 'fred': ['.*', '@runner']}}}}}

def test_rest_auth_config():
    if False:
        i = 10
        return i + 15
    ret = rest._rest_auth_setup()
    assert ret == 'https://test_url/rest'

def test_fetch_call_failed():
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.http.query', MagicMock(return_value={'status': 401})):
        ret = rest.fetch('foo', None)
        assert ret is False

def test_fetch_call_success_dict_none():
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.http.query', MagicMock(return_value={'status': 200, 'dict': None})):
        ret = rest.fetch('foo', None)
        assert ret == []

def test_fetch_call_success_dict_acl():
    if False:
        while True:
            i = 10
    with patch('salt.utils.http.query', MagicMock(return_value={'status': 200, 'dict': {'foo': ['@wheel']}})):
        ret = rest.fetch('foo', None)
        assert ret == {'foo': ['@wheel']}

def test_auth_nopass():
    if False:
        for i in range(10):
            print('nop')
    ret = rest.auth('foo', None)
    assert ret is False

def test_auth_nouser():
    if False:
        for i in range(10):
            print('nop')
    ret = rest.auth(None, 'foo')
    assert ret is False

def test_auth_nouserandpass():
    if False:
        i = 10
        return i + 15
    ret = rest.auth(None, None)
    assert ret is False

def test_auth_ok():
    if False:
        return 10
    with patch('salt.utils.http.query', MagicMock(return_value={'status': 200, 'dict': ['@wheel']})):
        ret = rest.auth('foo', None)
        assert ret is True

def test_acl_without_merge():
    if False:
        while True:
            i = 10
    ret = rest.acl('fred', password='password')
    assert ret == ['.*', '@runner']

def test_acl_unauthorized():
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.http.query', MagicMock(return_value={'status': 400})):
        ret = rest.acl('foo', password='password')
        assert ret is None

def test_acl_no_merge():
    if False:
        print('Hello World!')
    with patch('salt.utils.http.query', MagicMock(return_value={'status': 200, 'dict': None})):
        ret = rest.acl('fred', password='password')
        assert ret == ['.*', '@runner']

def test_acl_merge():
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.http.query', MagicMock(return_value={'status': 200, 'dict': ['@wheel']})):
        ret = rest.acl('fred', password='password')
        assert ret == ['.*', '@runner', '@wheel']