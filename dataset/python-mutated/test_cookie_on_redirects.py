import pytest
from .utils import http

def _stringify(fixture):
    if False:
        i = 10
        return i + 15
    return fixture + ''

@pytest.mark.parametrize('instance', [pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('remote_httpbin')])
def test_explicit_user_set_cookie(httpbin, instance):
    if False:
        print('Hello World!')
    r = http('--follow', httpbin + '/redirect-to', f'url=={_stringify(instance)}/cookies', 'Cookie:a=b')
    assert r.json == {'cookies': {}}

@pytest.mark.parametrize('instance', [pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('remote_httpbin')])
def test_explicit_user_set_cookie_in_session(tmp_path, httpbin, instance):
    if False:
        while True:
            i = 10
    r = http('--follow', '--session', str(tmp_path / 'session.json'), httpbin + '/redirect-to', f'url=={_stringify(instance)}/cookies', 'Cookie:a=b')
    assert r.json == {'cookies': {'a': 'b'}}

@pytest.mark.parametrize('instance', [pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('remote_httpbin')])
def test_saved_user_set_cookie_in_session(tmp_path, httpbin, instance):
    if False:
        for i in range(10):
            print('nop')
    http('--follow', '--session', str(tmp_path / 'session.json'), httpbin + '/get', 'Cookie:a=b')
    r = http('--follow', '--session', str(tmp_path / 'session.json'), httpbin + '/redirect-to', f'url=={_stringify(instance)}/cookies')
    assert r.json == {'cookies': {'a': 'b'}}

@pytest.mark.parametrize('instance', [pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('remote_httpbin')])
@pytest.mark.parametrize('session', [True, False])
def test_explicit_user_set_headers(httpbin, tmp_path, instance, session):
    if False:
        i = 10
        return i + 15
    session_args = []
    if session:
        session_args.extend(['--session', str(tmp_path / 'session.json')])
    r = http('--follow', *session_args, httpbin + '/redirect-to', f'url=={_stringify(instance)}/get', 'X-Custom-Header:value')
    assert 'X-Custom-Header' in r.json['headers']

@pytest.mark.parametrize('session', [True, False])
def test_server_set_cookie_on_redirect_same_domain(tmp_path, httpbin, session):
    if False:
        while True:
            i = 10
    session_args = []
    if session:
        session_args.extend(['--session', str(tmp_path / 'session.json')])
    r = http('--follow', *session_args, httpbin + '/cookies/set/a/b')
    assert r.json['cookies'] == {'a': 'b'}

@pytest.mark.parametrize('session', [True, False])
def test_server_set_cookie_on_redirect_different_domain(tmp_path, http_server, httpbin, session):
    if False:
        print('Hello World!')
    session_args = []
    if session:
        session_args.extend(['--session', str(tmp_path / 'session.json')])
    r = http('--follow', *session_args, http_server + '/cookies/set-and-redirect', f"X-Redirect-To:{httpbin + '/cookies'}", 'X-Cookies:a=b')
    assert r.json['cookies'] == {'a': 'b'}

def test_saved_session_cookies_on_same_domain(tmp_path, httpbin):
    if False:
        i = 10
        return i + 15
    http('--session', str(tmp_path / 'session.json'), httpbin + '/cookies/set/a/b')
    r = http('--session', str(tmp_path / 'session.json'), httpbin + '/cookies')
    assert r.json == {'cookies': {'a': 'b'}}

def test_saved_session_cookies_on_different_domain(tmp_path, httpbin, remote_httpbin):
    if False:
        i = 10
        return i + 15
    http('--session', str(tmp_path / 'session.json'), httpbin + '/cookies/set/a/b')
    r = http('--session', str(tmp_path / 'session.json'), remote_httpbin + '/cookies')
    assert r.json == {'cookies': {}}

@pytest.mark.parametrize('initial_domain, first_request_domain, second_request_domain, expect_cookies', [(pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('httpbin'), True), (pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('remote_httpbin'), pytest.lazy_fixture('remote_httpbin'), False), (pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('remote_httpbin'), False), (pytest.lazy_fixture('httpbin'), pytest.lazy_fixture('remote_httpbin'), pytest.lazy_fixture('httpbin'), True)])
def test_saved_session_cookies_on_redirect(tmp_path, initial_domain, first_request_domain, second_request_domain, expect_cookies):
    if False:
        i = 10
        return i + 15
    http('--session', str(tmp_path / 'session.json'), initial_domain + '/cookies/set/a/b')
    r = http('--session', str(tmp_path / 'session.json'), '--follow', first_request_domain + '/redirect-to', f'url=={_stringify(second_request_domain)}/cookies')
    if expect_cookies:
        expected_data = {'cookies': {'a': 'b'}}
    else:
        expected_data = {'cookies': {}}
    assert r.json == expected_data

def test_saved_session_cookie_pool(tmp_path, httpbin, remote_httpbin):
    if False:
        i = 10
        return i + 15
    http('--session', str(tmp_path / 'session.json'), httpbin + '/cookies/set/a/b')
    http('--session', str(tmp_path / 'session.json'), remote_httpbin + '/cookies/set/a/c')
    http('--session', str(tmp_path / 'session.json'), remote_httpbin + '/cookies/set/b/d')
    response = http('--session', str(tmp_path / 'session.json'), httpbin + '/cookies')
    assert response.json['cookies'] == {'a': 'b'}
    response = http('--session', str(tmp_path / 'session.json'), remote_httpbin + '/cookies')
    assert response.json['cookies'] == {'a': 'c', 'b': 'd'}