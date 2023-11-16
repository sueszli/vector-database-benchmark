"""
Various encoding handling related tests.

"""
import pytest
import responses
from charset_normalizer.constant import TOO_SMALL_SEQUENCE
from httpie.cli.constants import PRETTY_MAP
from httpie.encoding import UTF8
from .utils import http, HTTP_OK, DUMMY_URL, MockEnvironment
from .fixtures import UNICODE
CHARSET_TEXT_PAIRS = [('big5', '卷首卷首卷首卷首卷卷首卷首卷首卷首卷首卷首卷首卷首卷首卷首卷首卷首卷首'), ('windows-1250', 'Všichni lidé jsou si rovni. Všichni lidé jsou si rovni.'), (UTF8, 'Všichni lidé jsou si rovni. Všichni lidé jsou si rovni.')]

def test_charset_text_pairs():
    if False:
        while True:
            i = 10
    for (charset, text) in CHARSET_TEXT_PAIRS:
        assert len(text) > TOO_SMALL_SEQUENCE
        if charset != UTF8:
            with pytest.raises(UnicodeDecodeError):
                assert text != text.encode(charset).decode(UTF8)

def test_unicode_headers(httpbin):
    if False:
        return 10
    r = http(httpbin.url + '/headers', f'Test:{UNICODE}')
    assert HTTP_OK in r

def test_unicode_headers_verbose(httpbin):
    if False:
        for i in range(10):
            print('nop')
    r = http('--verbose', httpbin.url + '/headers', f'Test:{UNICODE}')
    assert HTTP_OK in r
    assert UNICODE in r

def test_unicode_raw(httpbin):
    if False:
        print('Hello World!')
    r = http('--raw', f'test {UNICODE}', 'POST', httpbin.url + '/post')
    assert HTTP_OK in r
    assert r.json['data'] == f'test {UNICODE}'

def test_unicode_raw_verbose(httpbin):
    if False:
        i = 10
        return i + 15
    r = http('--verbose', '--raw', f'test {UNICODE}', 'POST', httpbin.url + '/post')
    assert HTTP_OK in r
    assert UNICODE in r

def test_unicode_form_item(httpbin):
    if False:
        i = 10
        return i + 15
    r = http('--form', 'POST', httpbin.url + '/post', f'test={UNICODE}')
    assert HTTP_OK in r
    assert r.json['form'] == {'test': UNICODE}

def test_unicode_form_item_verbose(httpbin):
    if False:
        i = 10
        return i + 15
    r = http('--verbose', '--form', 'POST', httpbin.url + '/post', f'test={UNICODE}')
    assert HTTP_OK in r
    assert UNICODE in r

def test_unicode_json_item(httpbin):
    if False:
        for i in range(10):
            print('nop')
    r = http('--json', 'POST', httpbin.url + '/post', f'test={UNICODE}')
    assert HTTP_OK in r
    assert r.json['json'] == {'test': UNICODE}

def test_unicode_json_item_verbose(httpbin):
    if False:
        for i in range(10):
            print('nop')
    r = http('--verbose', '--json', 'POST', httpbin.url + '/post', f'test={UNICODE}')
    assert HTTP_OK in r
    assert UNICODE in r

def test_unicode_raw_json_item(httpbin):
    if False:
        i = 10
        return i + 15
    r = http('--json', 'POST', httpbin.url + '/post', f'test:={{ "{UNICODE}" : [ "{UNICODE}" ] }}')
    assert HTTP_OK in r
    assert r.json['json'] == {'test': {UNICODE: [UNICODE]}}

def test_unicode_raw_json_item_verbose(httpbin):
    if False:
        i = 10
        return i + 15
    r = http('--json', 'POST', httpbin.url + '/post', f'test:={{ "{UNICODE}" : [ "{UNICODE}" ] }}')
    assert HTTP_OK in r
    assert r.json['json'] == {'test': {UNICODE: [UNICODE]}}

def test_unicode_url_query_arg_item(httpbin):
    if False:
        while True:
            i = 10
    r = http(httpbin.url + '/get', f'test=={UNICODE}')
    assert HTTP_OK in r
    assert r.json['args'] == {'test': UNICODE}, r

def test_unicode_url_query_arg_item_verbose(httpbin):
    if False:
        i = 10
        return i + 15
    r = http('--verbose', httpbin.url + '/get', f'test=={UNICODE}')
    assert HTTP_OK in r
    assert UNICODE in r

def test_unicode_url(httpbin):
    if False:
        return 10
    r = http(f'{httpbin.url}/get?test={UNICODE}')
    assert HTTP_OK in r
    assert r.json['args'] == {'test': UNICODE}

def test_unicode_url_verbose(httpbin):
    if False:
        i = 10
        return i + 15
    r = http('--verbose', f'{httpbin.url}/get?test={UNICODE}')
    assert HTTP_OK in r
    assert r.json['args'] == {'test': UNICODE}

def test_unicode_basic_auth(httpbin):
    if False:
        print('Hello World!')
    http('--verbose', '--auth', f'test:{UNICODE}', f'{httpbin.url}/basic-auth/test/{UNICODE}')

def test_unicode_digest_auth(httpbin):
    if False:
        i = 10
        return i + 15
    http('--auth-type=digest', '--auth', f'test:{UNICODE}', f'{httpbin.url}/digest-auth/auth/test/{UNICODE}')

@pytest.mark.parametrize('charset, text', CHARSET_TEXT_PAIRS)
@responses.activate
def test_terminal_output_response_charset_detection(text, charset):
    if False:
        for i in range(10):
            print('nop')
    responses.add(method=responses.POST, url=DUMMY_URL, body=text.encode(charset), content_type='text/plain')
    r = http('--form', 'POST', DUMMY_URL)
    assert text in r

@pytest.mark.parametrize('charset, text', CHARSET_TEXT_PAIRS)
@responses.activate
def test_terminal_output_response_content_type_charset(charset, text):
    if False:
        for i in range(10):
            print('nop')
    responses.add(method=responses.POST, url=DUMMY_URL, body=text.encode(charset), content_type=f'text/plain; charset={charset}')
    r = http('--form', 'POST', DUMMY_URL)
    assert text in r

@pytest.mark.parametrize('charset, text', CHARSET_TEXT_PAIRS)
@pytest.mark.parametrize('pretty', PRETTY_MAP.keys())
@responses.activate
def test_terminal_output_response_content_type_charset_with_stream(charset, text, pretty):
    if False:
        while True:
            i = 10
    responses.add(method=responses.GET, url=DUMMY_URL, body=f'<?xml version="1.0"?>\n<c>{text}</c>'.encode(charset), stream=True, content_type=f'text/xml; charset={charset.upper()}')
    r = http('--pretty', pretty, '--stream', DUMMY_URL)
    assert text in r

@pytest.mark.parametrize('charset, text', CHARSET_TEXT_PAIRS)
@pytest.mark.parametrize('pretty', PRETTY_MAP.keys())
@responses.activate
def test_terminal_output_response_charset_override(charset, text, pretty):
    if False:
        for i in range(10):
            print('nop')
    responses.add(responses.GET, DUMMY_URL, body=text.encode(charset), content_type='text/plain; charset=utf-8')
    args = ['--pretty', pretty, DUMMY_URL]
    if charset != UTF8:
        r = http(*args)
        assert text not in r
    r = http('--response-charset', charset, *args)
    assert text in r

@pytest.mark.parametrize('charset, text', CHARSET_TEXT_PAIRS)
def test_terminal_output_request_content_type_charset(charset, text):
    if False:
        print('Hello World!')
    r = http('--offline', DUMMY_URL, f'Content-Type: text/plain; charset={charset.upper()}', env=MockEnvironment(stdin=text.encode(charset), stdin_isatty=False))
    assert text in r

@pytest.mark.parametrize('charset, text', CHARSET_TEXT_PAIRS)
def test_terminal_output_request_charset_detection(charset, text):
    if False:
        return 10
    r = http('--offline', DUMMY_URL, 'Content-Type: text/plain', env=MockEnvironment(stdin=text.encode(charset), stdin_isatty=False))
    assert text in r