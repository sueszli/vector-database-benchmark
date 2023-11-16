from __future__ import annotations
import io
import socket
import sys
import http.client
import urllib.error
from http.cookiejar import Cookie
from ansible.module_utils.urls import fetch_url, ConnectionError
import pytest
from unittest.mock import MagicMock

class AnsibleModuleExit(Exception):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.args = args
        self.kwargs = kwargs

class ExitJson(AnsibleModuleExit):
    pass

class FailJson(AnsibleModuleExit):
    pass

@pytest.fixture
def open_url_mock(mocker):
    if False:
        return 10
    return mocker.patch('ansible.module_utils.urls.open_url')

@pytest.fixture
def fake_ansible_module():
    if False:
        print('Hello World!')
    return FakeAnsibleModule()

class FakeAnsibleModule:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.params = {}
        self.tmpdir = None

    def exit_json(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise ExitJson(*args, **kwargs)

    def fail_json(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise FailJson(*args, **kwargs)

def test_fetch_url(open_url_mock, fake_ansible_module):
    if False:
        print('Hello World!')
    (r, info) = fetch_url(fake_ansible_module, 'http://ansible.com/')
    (dummy, kwargs) = open_url_mock.call_args
    open_url_mock.assert_called_once_with('http://ansible.com/', client_cert=None, client_key=None, cookies=kwargs['cookies'], data=None, follow_redirects='urllib2', force=False, force_basic_auth='', headers=None, http_agent='ansible-httpget', last_mod_time=None, method=None, timeout=10, url_password='', url_username='', use_proxy=True, validate_certs=True, use_gssapi=False, unix_socket=None, ca_path=None, unredirected_headers=None, decompress=True, ciphers=None, use_netrc=True)

def test_fetch_url_params(open_url_mock, fake_ansible_module):
    if False:
        i = 10
        return i + 15
    fake_ansible_module.params = {'validate_certs': False, 'url_username': 'user', 'url_password': 'passwd', 'http_agent': 'ansible-test', 'force_basic_auth': True, 'follow_redirects': 'all', 'client_cert': 'client.pem', 'client_key': 'client.key'}
    (r, info) = fetch_url(fake_ansible_module, 'http://ansible.com/')
    (dummy, kwargs) = open_url_mock.call_args
    open_url_mock.assert_called_once_with('http://ansible.com/', client_cert='client.pem', client_key='client.key', cookies=kwargs['cookies'], data=None, follow_redirects='all', force=False, force_basic_auth=True, headers=None, http_agent='ansible-test', last_mod_time=None, method=None, timeout=10, url_password='passwd', url_username='user', use_proxy=True, validate_certs=False, use_gssapi=False, unix_socket=None, ca_path=None, unredirected_headers=None, decompress=True, ciphers=None, use_netrc=True)

def test_fetch_url_cookies(mocker, fake_ansible_module):
    if False:
        i = 10
        return i + 15

    def make_cookies(*args, **kwargs):
        if False:
            while True:
                i = 10
        cookies = kwargs['cookies']
        r = MagicMock()
        r.headers = http.client.HTTPMessage()
        add_header = r.headers.add_header
        r.info.return_value = r.headers
        for (name, value) in (('Foo', 'bar'), ('Baz', 'qux')):
            cookie = Cookie(version=0, name=name, value=value, port=None, port_specified=False, domain='ansible.com', domain_specified=True, domain_initial_dot=False, path='/', path_specified=True, secure=False, expires=None, discard=False, comment=None, comment_url=None, rest=None)
            cookies.set_cookie(cookie)
            add_header('Set-Cookie', '%s=%s' % (name, value))
        return r
    mocker = mocker.patch('ansible.module_utils.urls.open_url', new=make_cookies)
    (r, info) = fetch_url(fake_ansible_module, 'http://ansible.com/')
    assert info['cookies'] == {'Baz': 'qux', 'Foo': 'bar'}
    if sys.version_info < (3, 11):
        assert info['cookies_string'] == 'Baz=qux; Foo=bar'
    else:
        assert info['cookies_string'] == 'Foo=bar; Baz=qux'
    assert info['set-cookie'] == 'Foo=bar, Baz=qux'

def test_fetch_url_connectionerror(open_url_mock, fake_ansible_module):
    if False:
        return 10
    open_url_mock.side_effect = ConnectionError('TESTS')
    with pytest.raises(FailJson) as excinfo:
        fetch_url(fake_ansible_module, 'http://ansible.com/')
    assert excinfo.value.kwargs['msg'] == 'TESTS'
    assert 'http://ansible.com/' == excinfo.value.kwargs['url']
    assert excinfo.value.kwargs['status'] == -1
    open_url_mock.side_effect = ValueError('TESTS')
    with pytest.raises(FailJson) as excinfo:
        fetch_url(fake_ansible_module, 'http://ansible.com/')
    assert excinfo.value.kwargs['msg'] == 'TESTS'
    assert 'http://ansible.com/' == excinfo.value.kwargs['url']
    assert excinfo.value.kwargs['status'] == -1

def test_fetch_url_httperror(open_url_mock, fake_ansible_module):
    if False:
        while True:
            i = 10
    open_url_mock.side_effect = urllib.error.HTTPError('http://ansible.com/', 500, 'Internal Server Error', {'Content-Type': 'application/json'}, io.StringIO('TESTS'))
    (r, info) = fetch_url(fake_ansible_module, 'http://ansible.com/')
    assert info == {'msg': 'HTTP Error 500: Internal Server Error', 'body': 'TESTS', 'status': 500, 'url': 'http://ansible.com/', 'content-type': 'application/json'}

def test_fetch_url_urlerror(open_url_mock, fake_ansible_module):
    if False:
        i = 10
        return i + 15
    open_url_mock.side_effect = urllib.error.URLError('TESTS')
    (r, info) = fetch_url(fake_ansible_module, 'http://ansible.com/')
    assert info == {'msg': 'Request failed: <urlopen error TESTS>', 'status': -1, 'url': 'http://ansible.com/'}

def test_fetch_url_socketerror(open_url_mock, fake_ansible_module):
    if False:
        i = 10
        return i + 15
    open_url_mock.side_effect = socket.error('TESTS')
    (r, info) = fetch_url(fake_ansible_module, 'http://ansible.com/')
    assert info == {'msg': 'Connection failure: TESTS', 'status': -1, 'url': 'http://ansible.com/'}

def test_fetch_url_exception(open_url_mock, fake_ansible_module):
    if False:
        return 10
    open_url_mock.side_effect = Exception('TESTS')
    (r, info) = fetch_url(fake_ansible_module, 'http://ansible.com/')
    exception = info.pop('exception')
    assert info == {'msg': 'An unknown error occurred: TESTS', 'status': -1, 'url': 'http://ansible.com/'}
    assert 'Exception: TESTS' in exception

def test_fetch_url_badstatusline(open_url_mock, fake_ansible_module):
    if False:
        i = 10
        return i + 15
    open_url_mock.side_effect = http.client.BadStatusLine('TESTS')
    (r, info) = fetch_url(fake_ansible_module, 'http://ansible.com/')
    assert info == {'msg': 'Connection failure: connection was closed before a valid response was received: TESTS', 'status': -1, 'url': 'http://ansible.com/'}