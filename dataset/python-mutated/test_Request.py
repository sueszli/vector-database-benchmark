from __future__ import annotations
import datetime
import os
import urllib.request
import http.client
from ansible.module_utils.urls import Request, open_url, cookiejar, UnixHTTPHandler, UnixHTTPSConnection
from ansible.module_utils.urls import HTTPRedirectHandler
import pytest
from unittest.mock import call
import ssl

@pytest.fixture
def urlopen_mock(mocker):
    if False:
        for i in range(10):
            print('nop')
    return mocker.patch('ansible.module_utils.urls.urllib.request.urlopen')

@pytest.fixture
def install_opener_mock(mocker):
    if False:
        return 10
    return mocker.patch('ansible.module_utils.urls.urllib.request.install_opener')

def test_Request_fallback(urlopen_mock, install_opener_mock, mocker):
    if False:
        while True:
            i = 10
    here = os.path.dirname(__file__)
    pem = os.path.join(here, 'fixtures/client.pem')
    client_key = os.path.join(here, 'fixtures/client.key')
    cookies = cookiejar.CookieJar()
    request = Request(headers={'foo': 'bar'}, use_proxy=False, force=True, timeout=100, validate_certs=False, url_username='user', url_password='passwd', http_agent='ansible-tests', force_basic_auth=True, follow_redirects='all', client_cert=pem, client_key=client_key, cookies=cookies, unix_socket='/foo/bar/baz.sock', ca_path=pem, ciphers=['ECDHE-RSA-AES128-SHA256'], use_netrc=True)
    fallback_mock = mocker.spy(request, '_fallback')
    r = request.open('GET', 'https://ansible.com')
    calls = [call(None, False), call(None, True), call(None, 100), call(None, False), call(None, 'user'), call(None, 'passwd'), call(None, 'ansible-tests'), call(None, True), call(None, 'all'), call(None, pem), call(None, client_key), call(None, cookies), call(None, '/foo/bar/baz.sock'), call(None, pem), call(None, None), call(None, True), call(None, ['ECDHE-RSA-AES128-SHA256']), call(None, True), call(None, None)]
    fallback_mock.assert_has_calls(calls)
    assert fallback_mock.call_count == 19
    args = urlopen_mock.call_args[0]
    assert args[1] is None
    assert args[2] == 100
    req = args[0]
    assert req.headers == {'Authorization': b'Basic dXNlcjpwYXNzd2Q=', 'Cache-control': 'no-cache', 'Foo': 'bar', 'User-agent': 'ansible-tests'}
    assert req.data is None
    assert req.get_method() == 'GET'

def test_Request_open(urlopen_mock, install_opener_mock):
    if False:
        i = 10
        return i + 15
    r = Request().open('GET', 'https://ansible.com/')
    args = urlopen_mock.call_args[0]
    assert args[1] is None
    assert args[2] == 10
    req = args[0]
    assert req.headers == {}
    assert req.data is None
    assert req.get_method() == 'GET'
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    expected_handlers = (HTTPRedirectHandler(),)
    found_handlers = []
    for handler in handlers:
        if handler.__class__.__name__ == 'HTTPRedirectHandler':
            found_handlers.append(handler)
    assert len(found_handlers) == len(expected_handlers)

def test_Request_open_unix_socket(urlopen_mock, install_opener_mock):
    if False:
        while True:
            i = 10
    r = Request().open('GET', 'http://ansible.com/', unix_socket='/foo/bar/baz.sock')
    args = urlopen_mock.call_args[0]
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    found_handlers = []
    for handler in handlers:
        if isinstance(handler, UnixHTTPHandler):
            found_handlers.append(handler)
    assert len(found_handlers) == 1

def test_Request_open_https_unix_socket(urlopen_mock, install_opener_mock, mocker):
    if False:
        print('Hello World!')
    do_open = mocker.patch.object(urllib.request.HTTPSHandler, 'do_open')
    r = Request().open('GET', 'https://ansible.com/', unix_socket='/foo/bar/baz.sock')
    args = urlopen_mock.call_args[0]
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    found_handlers = []
    for handler in handlers:
        if isinstance(handler, urllib.request.HTTPSHandler):
            found_handlers.append(handler)
    assert len(found_handlers) == 1
    found_handlers[0].https_open(None)
    args = do_open.call_args[0]
    cls = args[0]
    assert isinstance(cls, UnixHTTPSConnection)

def test_Request_open_ftp(urlopen_mock, install_opener_mock, mocker):
    if False:
        return 10
    mocker.patch('ansible.module_utils.urls.ParseResultDottedDict.as_list', side_effect=AssertionError)
    r = Request().open('GET', 'ftp://foo@ansible.com/')

def test_Request_open_headers(urlopen_mock, install_opener_mock):
    if False:
        for i in range(10):
            print('nop')
    r = Request().open('GET', 'http://ansible.com/', headers={'Foo': 'bar'})
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert req.headers == {'Foo': 'bar'}

def test_Request_open_username(urlopen_mock, install_opener_mock):
    if False:
        print('Hello World!')
    r = Request().open('GET', 'http://ansible.com/', url_username='user')
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    expected_handlers = (urllib.request.HTTPBasicAuthHandler, urllib.request.HTTPDigestAuthHandler)
    found_handlers = []
    for handler in handlers:
        if isinstance(handler, expected_handlers):
            found_handlers.append(handler)
    assert len(found_handlers) == 2
    assert found_handlers[0].passwd.passwd[None] == {(('ansible.com', '/'),): ('user', None)}

def test_Request_open_username_in_url(urlopen_mock, install_opener_mock):
    if False:
        i = 10
        return i + 15
    r = Request().open('GET', 'http://user2@ansible.com/')
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    expected_handlers = (urllib.request.HTTPBasicAuthHandler, urllib.request.HTTPDigestAuthHandler)
    found_handlers = []
    for handler in handlers:
        if isinstance(handler, expected_handlers):
            found_handlers.append(handler)
    assert found_handlers[0].passwd.passwd[None] == {(('ansible.com', '/'),): ('user2', '')}

def test_Request_open_username_force_basic(urlopen_mock, install_opener_mock):
    if False:
        return 10
    r = Request().open('GET', 'http://ansible.com/', url_username='user', url_password='passwd', force_basic_auth=True)
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    expected_handlers = (urllib.request.HTTPBasicAuthHandler, urllib.request.HTTPDigestAuthHandler)
    found_handlers = []
    for handler in handlers:
        if isinstance(handler, expected_handlers):
            found_handlers.append(handler)
    assert len(found_handlers) == 0
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert req.headers.get('Authorization') == b'Basic dXNlcjpwYXNzd2Q='

def test_Request_open_auth_in_netloc(urlopen_mock, install_opener_mock):
    if False:
        for i in range(10):
            print('nop')
    r = Request().open('GET', 'http://user:passwd@ansible.com/')
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert req.get_full_url() == 'http://ansible.com/'
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    expected_handlers = (urllib.request.HTTPBasicAuthHandler, urllib.request.HTTPDigestAuthHandler)
    found_handlers = []
    for handler in handlers:
        if isinstance(handler, expected_handlers):
            found_handlers.append(handler)
    assert len(found_handlers) == 2

def test_Request_open_netrc(urlopen_mock, install_opener_mock, monkeypatch):
    if False:
        print('Hello World!')
    here = os.path.dirname(__file__)
    monkeypatch.setenv('NETRC', os.path.join(here, 'fixtures/netrc'))
    r = Request().open('GET', 'http://ansible.com/')
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert req.headers.get('Authorization') == b'Basic dXNlcjpwYXNzd2Q='
    r = Request().open('GET', 'http://foo.ansible.com/')
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert 'Authorization' not in req.headers
    monkeypatch.setenv('NETRC', os.path.join(here, 'fixtures/netrc.nonexistant'))
    r = Request().open('GET', 'http://ansible.com/')
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert 'Authorization' not in req.headers

def test_Request_open_no_proxy(urlopen_mock, install_opener_mock, mocker):
    if False:
        i = 10
        return i + 15
    build_opener_mock = mocker.patch('ansible.module_utils.urls.urllib.request.build_opener')
    r = Request().open('GET', 'http://ansible.com/', use_proxy=False)
    handlers = build_opener_mock.call_args[0]
    found_handlers = []
    for handler in handlers:
        if isinstance(handler, urllib.request.ProxyHandler):
            found_handlers.append(handler)
    assert len(found_handlers) == 1

def test_Request_open_no_validate_certs(urlopen_mock, install_opener_mock, mocker):
    if False:
        return 10
    do_open = mocker.patch.object(urllib.request.HTTPSHandler, 'do_open')
    r = Request().open('GET', 'https://ansible.com/', validate_certs=False)
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    ssl_handler = None
    for handler in handlers:
        if isinstance(handler, urllib.request.HTTPSHandler):
            ssl_handler = handler
            break
    assert ssl_handler is not None
    ssl_handler.https_open(None)
    args = do_open.call_args[0]
    cls = args[0]
    assert cls is http.client.HTTPSConnection
    context = ssl_handler._context
    if ssl.OP_NO_SSLv2:
        assert context.options & ssl.OP_NO_SSLv2
    assert context.options & ssl.OP_NO_SSLv3
    assert context.verify_mode == ssl.CERT_NONE
    assert context.check_hostname is False

def test_Request_open_client_cert(urlopen_mock, install_opener_mock, mocker):
    if False:
        i = 10
        return i + 15
    load_cert_chain = mocker.patch.object(ssl.SSLContext, 'load_cert_chain')
    here = os.path.dirname(__file__)
    client_cert = os.path.join(here, 'fixtures/client.pem')
    client_key = os.path.join(here, 'fixtures/client.key')
    r = Request().open('GET', 'https://ansible.com/', client_cert=client_cert, client_key=client_key)
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    ssl_handler = None
    for handler in handlers:
        if isinstance(handler, urllib.request.HTTPSHandler):
            ssl_handler = handler
            break
    assert ssl_handler is not None
    load_cert_chain.assert_called_once_with(client_cert, keyfile=client_key)

def test_Request_open_cookies(urlopen_mock, install_opener_mock):
    if False:
        while True:
            i = 10
    r = Request().open('GET', 'https://ansible.com/', cookies=cookiejar.CookieJar())
    opener = install_opener_mock.call_args[0][0]
    handlers = opener.handlers
    cookies_handler = None
    for handler in handlers:
        if isinstance(handler, urllib.request.HTTPCookieProcessor):
            cookies_handler = handler
            break
    assert cookies_handler is not None

def test_Request_open_invalid_method(urlopen_mock, install_opener_mock):
    if False:
        print('Hello World!')
    r = Request().open('UNKNOWN', 'https://ansible.com/')
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert req.data is None
    assert req.get_method() == 'UNKNOWN'

def test_Request_open_user_agent(urlopen_mock, install_opener_mock):
    if False:
        while True:
            i = 10
    r = Request().open('GET', 'https://ansible.com/', http_agent='ansible-tests')
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert req.headers.get('User-agent') == 'ansible-tests'

def test_Request_open_force(urlopen_mock, install_opener_mock):
    if False:
        i = 10
        return i + 15
    r = Request().open('GET', 'https://ansible.com/', force=True, last_mod_time=datetime.datetime.now())
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert req.headers.get('Cache-control') == 'no-cache'
    assert 'If-modified-since' not in req.headers

def test_Request_open_last_mod(urlopen_mock, install_opener_mock):
    if False:
        return 10
    now = datetime.datetime.now()
    r = Request().open('GET', 'https://ansible.com/', last_mod_time=now)
    args = urlopen_mock.call_args[0]
    req = args[0]
    assert req.headers.get('If-modified-since') == now.strftime('%a, %d %b %Y %H:%M:%S GMT')

def test_Request_open_headers_not_dict(urlopen_mock, install_opener_mock):
    if False:
        return 10
    with pytest.raises(ValueError):
        Request().open('GET', 'https://ansible.com/', headers=['bob'])

def test_Request_init_headers_not_dict(urlopen_mock, install_opener_mock):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        Request(headers=['bob'])

@pytest.mark.parametrize('method,kwargs', [('get', {}), ('options', {}), ('head', {}), ('post', {'data': None}), ('put', {'data': None}), ('patch', {'data': None}), ('delete', {})])
def test_methods(method, kwargs, mocker):
    if False:
        return 10
    expected = method.upper()
    open_mock = mocker.patch('ansible.module_utils.urls.Request.open')
    request = Request()
    getattr(request, method)('https://ansible.com')
    open_mock.assert_called_once_with(expected, 'https://ansible.com', **kwargs)

def test_open_url(urlopen_mock, install_opener_mock, mocker):
    if False:
        while True:
            i = 10
    req_mock = mocker.patch('ansible.module_utils.urls.Request.open')
    open_url('https://ansible.com/')
    req_mock.assert_called_once_with('GET', 'https://ansible.com/', data=None, headers=None, use_proxy=True, force=False, last_mod_time=None, timeout=10, validate_certs=True, url_username=None, url_password=None, http_agent=None, force_basic_auth=False, follow_redirects='urllib2', client_cert=None, client_key=None, cookies=None, use_gssapi=False, unix_socket=None, ca_path=None, unredirected_headers=None, decompress=True, ciphers=None, use_netrc=True)