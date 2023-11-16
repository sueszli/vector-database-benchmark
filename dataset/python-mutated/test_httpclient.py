import re
import pytest
from mopidy import httpclient

@pytest.mark.parametrize(('config', 'expected'), [({}, None), ({'hostname': ''}, None), ({'hostname': 'proxy.lan'}, 'http://proxy.lan:80'), ({'scheme': None, 'hostname': 'proxy.lan'}, 'http://proxy.lan:80'), ({'scheme': 'https', 'hostname': 'proxy.lan'}, 'https://proxy.lan:80'), ({'username': 'user', 'hostname': 'proxy.lan'}, 'http://proxy.lan:80'), ({'password': 'pass', 'hostname': 'proxy.lan'}, 'http://proxy.lan:80'), ({'hostname': 'proxy.lan', 'port': 8080}, 'http://proxy.lan:8080'), ({'hostname': 'proxy.lan', 'port': -1}, 'http://proxy.lan:80'), ({'hostname': 'proxy.lan', 'port': None}, 'http://proxy.lan:80'), ({'hostname': 'proxy.lan', 'port': ''}, 'http://proxy.lan:80'), ({'username': 'user', 'password': 'pass', 'hostname': 'proxy.lan'}, 'http://user:pass@proxy.lan:80')])
def test_format_proxy(config, expected):
    if False:
        i = 10
        return i + 15
    assert httpclient.format_proxy(config) == expected

def test_format_proxy_without_auth():
    if False:
        print('Hello World!')
    config = {'username': 'user', 'password': 'pass', 'hostname': 'proxy.lan'}
    formated_proxy = httpclient.format_proxy(config, auth=False)
    assert formated_proxy == 'http://proxy.lan:80'

@pytest.mark.parametrize(('name', 'expected'), [(None, '^Mopidy/[^ ]+ CPython|/[^ ]+$'), ('Foo', '^Foo Mopidy/[^ ]+ CPython|/[^ ]+$'), ('Foo/1.2.3', '^Foo/1.2.3 Mopidy/[^ ]+ CPython|/[^ ]+$')])
def test_format_user_agent(name, expected):
    if False:
        i = 10
        return i + 15
    assert re.match(expected, httpclient.format_user_agent(name))