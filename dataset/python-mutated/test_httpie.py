"""High-level tests."""
import io
from unittest import mock
import pytest
import httpie
import httpie.__main__
from .fixtures import FILE_CONTENT, FILE_PATH
from httpie.cli.exceptions import ParseError
from httpie.context import Environment
from httpie.encoding import UTF8
from httpie.status import ExitStatus
from .utils import HTTP_OK, MockEnvironment, StdinBytesIO, http

def test_main_entry_point():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(Environment, 'stdin', io.StringIO()):
        assert httpie.__main__.main() == ExitStatus.ERROR.value

@mock.patch('httpie.core.main')
def test_main_entry_point_keyboard_interrupt(main):
    if False:
        i = 10
        return i + 15
    main.side_effect = KeyboardInterrupt()
    with mock.patch.object(Environment, 'stdin', io.StringIO()):
        assert httpie.__main__.main() == ExitStatus.ERROR_CTRL_C.value

def test_debug():
    if False:
        for i in range(10):
            print('nop')
    r = http('--debug')
    assert r.exit_status == ExitStatus.SUCCESS
    assert f'HTTPie {httpie.__version__}' in r.stderr

def test_help():
    if False:
        return 10
    r = http('--help', tolerate_error_exit_status=True)
    assert r.exit_status == ExitStatus.SUCCESS
    assert 'https://github.com/httpie/cli/issues' in r

def test_version():
    if False:
        while True:
            i = 10
    r = http('--version', tolerate_error_exit_status=True)
    assert r.exit_status == ExitStatus.SUCCESS
    assert httpie.__version__ == r.strip()

def test_GET(httpbin_both):
    if False:
        for i in range(10):
            print('nop')
    r = http('GET', httpbin_both + '/get')
    assert HTTP_OK in r

def test_path_dot_normalization():
    if False:
        for i in range(10):
            print('nop')
    r = http('--offline', 'example.org/../../etc/password', 'param==value')
    assert 'GET /etc/password?param=value' in r

def test_path_as_is():
    if False:
        while True:
            i = 10
    r = http('--offline', '--path-as-is', 'example.org/../../etc/password', 'param==value')
    assert 'GET /../../etc/password?param=value' in r

def test_DELETE(httpbin_both):
    if False:
        return 10
    r = http('DELETE', httpbin_both + '/delete')
    assert HTTP_OK in r

def test_PUT(httpbin_both):
    if False:
        while True:
            i = 10
    r = http('PUT', httpbin_both + '/put', 'foo=bar')
    assert HTTP_OK in r
    assert r.json['json']['foo'] == 'bar'

def test_POST_JSON_data(httpbin_both):
    if False:
        for i in range(10):
            print('nop')
    r = http('POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert r.json['json']['foo'] == 'bar'

def test_POST_form(httpbin_both):
    if False:
        print('Hello World!')
    r = http('--form', 'POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert '"foo": "bar"' in r

def test_POST_form_multiple_values(httpbin_both):
    if False:
        i = 10
        return i + 15
    r = http('--form', 'POST', httpbin_both + '/post', 'foo=bar', 'foo=baz')
    assert HTTP_OK in r
    assert r.json['form'] == {'foo': ['bar', 'baz']}

def test_POST_raw(httpbin_both):
    if False:
        while True:
            i = 10
    r = http('--raw', 'foo bar', 'POST', httpbin_both + '/post')
    assert HTTP_OK in r
    assert '"foo bar"' in r

def test_POST_stdin(httpbin_both):
    if False:
        i = 10
        return i + 15
    env = MockEnvironment(stdin=StdinBytesIO(FILE_PATH.read_bytes()), stdin_isatty=False)
    r = http('--form', 'POST', httpbin_both + '/post', env=env)
    assert HTTP_OK in r
    assert FILE_CONTENT in r

def test_POST_file(httpbin_both):
    if False:
        while True:
            i = 10
    r = http('--form', 'POST', httpbin_both + '/post', f'file@{FILE_PATH}')
    assert HTTP_OK in r
    assert FILE_CONTENT in r

def test_form_POST_file_redirected_stdin(httpbin):
    if False:
        print('Hello World!')
    '\n    <https://github.com/httpie/cli/issues/840>\n\n    '
    with open(FILE_PATH, encoding=UTF8):
        r = http('--form', 'POST', httpbin + '/post', f'file@{FILE_PATH}', tolerate_error_exit_status=True, env=MockEnvironment(stdin=StdinBytesIO(FILE_PATH.read_bytes()), stdin_isatty=False))
    assert r.exit_status == ExitStatus.ERROR
    assert 'cannot be mixed' in r.stderr

def test_raw_POST_key_values_supplied(httpbin):
    if False:
        for i in range(10):
            print('nop')
    r = http('--raw', 'foo bar', 'POST', httpbin + '/post', 'foo=bar', tolerate_error_exit_status=True)
    assert r.exit_status == ExitStatus.ERROR
    assert 'cannot be mixed' in r.stderr

def test_raw_POST_redirected_stdin(httpbin):
    if False:
        i = 10
        return i + 15
    r = http('--raw', 'foo bar', 'POST', httpbin + '/post', tolerate_error_exit_status=True, env=MockEnvironment(stdin='some=value', stdin_isatty=False))
    assert r.exit_status == ExitStatus.ERROR
    assert 'cannot be mixed' in r.stderr

def test_headers(httpbin_both):
    if False:
        return 10
    r = http('GET', httpbin_both + '/headers', 'Foo:bar')
    assert HTTP_OK in r
    assert '"User-Agent": "HTTPie' in r, r
    assert '"Foo": "bar"' in r

def test_headers_unset(httpbin_both):
    if False:
        i = 10
        return i + 15
    r = http('GET', httpbin_both + '/headers')
    assert 'Accept' in r.json['headers']
    r = http('GET', httpbin_both + '/headers', 'Accept:')
    assert 'Accept' not in r.json['headers']

@pytest.mark.skip('unimplemented')
def test_unset_host_header(httpbin_both):
    if False:
        for i in range(10):
            print('nop')
    r = http('GET', httpbin_both + '/headers')
    assert 'Host' in r.json['headers']
    r = http('GET', httpbin_both + '/headers', 'Host:')
    assert 'Host' not in r.json['headers']

def test_unset_useragent_header(httpbin_both):
    if False:
        for i in range(10):
            print('nop')
    r = http('GET', httpbin_both + '/headers')
    assert 'User-Agent' in r.json['headers']
    r = http('GET', httpbin_both + '/headers', 'User-Agent:')
    assert 'User-Agent' not in r.json['headers']

def test_headers_empty_value(httpbin_both):
    if False:
        for i in range(10):
            print('nop')
    r = http('GET', httpbin_both + '/headers')
    assert r.json['headers']['Accept']
    r = http('GET', httpbin_both + '/headers', 'Accept;')
    assert r.json['headers']['Accept'] == ''

def test_headers_empty_value_with_value_gives_error(httpbin):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ParseError):
        http('GET', httpbin + '/headers', 'Accept;SYNTAX_ERROR')

def test_headers_omit(httpbin_both):
    if False:
        for i in range(10):
            print('nop')
    r = http('GET', httpbin_both + '/headers', 'Accept:')
    assert 'Accept' not in r.json['headers']

def test_headers_multiple_omit(httpbin_both):
    if False:
        while True:
            i = 10
    r = http('GET', httpbin_both + '/headers', 'Foo:bar', 'Bar:baz', 'Foo:', 'Baz:quux')
    assert 'Foo' not in r.json['headers']
    assert r.json['headers']['Bar'] == 'baz'
    assert r.json['headers']['Baz'] == 'quux'

def test_headers_same_after_omit(httpbin_both):
    if False:
        while True:
            i = 10
    r = http('GET', httpbin_both + '/headers', 'Foo:bar', 'Foo:', 'Foo:quux')
    assert r.json['headers']['Foo'] == 'quux'

def test_headers_fully_omit(httpbin_both):
    if False:
        for i in range(10):
            print('nop')
    r = http('GET', httpbin_both + '/headers', 'Foo:bar', 'Foo:baz', 'Foo:')
    assert 'Foo' not in r.json['headers']

def test_headers_multiple_values(httpbin_both):
    if False:
        for i in range(10):
            print('nop')
    r = http('GET', httpbin_both + '/headers', 'Foo:bar', 'Foo:baz')
    assert r.json['headers']['Foo'] == 'bar,baz'

def test_headers_multiple_values_repeated(httpbin_both):
    if False:
        for i in range(10):
            print('nop')
    r = http('GET', httpbin_both + '/headers', 'Foo:bar', 'Foo:baz', 'Foo:bar')
    assert r.json['headers']['Foo'] == 'bar,baz,bar'

@pytest.mark.parametrize('headers, expected', [(['Foo;', 'Foo:bar'], ',bar'), (['Foo:bar', 'Foo;'], 'bar,'), (['Foo:bar', 'Foo;', 'Foo:baz'], 'bar,,baz')])
def test_headers_multiple_values_with_empty(httpbin_both, headers, expected):
    if False:
        while True:
            i = 10
    r = http('GET', httpbin_both + '/headers', *headers)
    assert r.json['headers']['Foo'] == expected

def test_headers_multiple_values_mixed(httpbin_both):
    if False:
        print('Hello World!')
    r = http('GET', httpbin_both + '/headers', 'Foo:bar', 'Vary:XXX', 'Foo:baz', 'Vary:YYY', 'Foo:quux')
    assert r.json['headers']['Vary'] == 'XXX,YYY'
    assert r.json['headers']['Foo'] == 'bar,baz,quux'

def test_headers_preserve_prepared_headers(httpbin_both):
    if False:
        i = 10
        return i + 15
    r = http('POST', httpbin_both + '/post', 'Content-Length:0', '--raw', 'foo')
    assert r.json['headers']['Content-Length'] == '3'

@pytest.mark.parametrize('pretty', ['format', 'none'])
def test_headers_multiple_headers_representation(httpbin_both, pretty):
    if False:
        i = 10
        return i + 15
    r = http('--offline', '--pretty', pretty, 'example.org', 'A:A', 'A:B', 'A:C', 'B:A', 'B:B', 'C:C', 'c:c')
    assert 'A: A' in r
    assert 'A: B' in r
    assert 'A: C' in r
    assert 'B: A' in r
    assert 'B: B' in r
    assert 'C: C' in r
    assert 'c: c' in r

def test_response_headers_multiple(http_server):
    if False:
        return 10
    r = http('GET', http_server + '/headers', 'Foo:bar', 'Foo:baz')
    assert 'Foo: bar' in r
    assert 'Foo: baz' in r

def test_response_headers_multiple_repeated(http_server):
    if False:
        print('Hello World!')
    r = http('GET', http_server + '/headers', 'Foo:bar', 'Foo:baz', 'Foo:bar')
    assert r.count('Foo: bar') == 2
    assert 'Foo: baz' in r

@pytest.mark.parametrize('pretty', ['format', 'none'])
def test_response_headers_multiple_representation(http_server, pretty):
    if False:
        while True:
            i = 10
    r = http('--pretty', pretty, http_server + '/headers', 'A:A', 'A:B', 'A:C', 'B:A', 'B:B', 'C:C', 'C:c')
    assert 'A: A' in r
    assert 'A: B' in r
    assert 'A: C' in r
    assert 'B: A' in r
    assert 'B: B' in r
    assert 'C: C' in r
    assert 'C: c' in r

def test_json_input_preserve_order(httpbin_both):
    if False:
        while True:
            i = 10
    r = http('PATCH', httpbin_both + '/patch', 'order:={"map":{"1":"first","2":"second"}}')
    assert HTTP_OK in r
    assert r.json['data'] == '{"order": {"map": {"1": "first", "2": "second"}}}'

@pytest.mark.parametrize('extra_args, expected_content_length', [(['Content-Length:0'], '0'), (['Content-Length:xxx'], 'xxx'), (['--raw=data'], '4'), (['query[param]=something'], '33')])
def test_options_content_length_preservation(httpbin, extra_args, expected_content_length):
    if False:
        return 10
    r = http('--offline', 'OPTIONS', httpbin + '/anything', *extra_args)
    assert f'Content-Length: {expected_content_length}' in r

@pytest.mark.parametrize('method', ['options', 'Options', 'OPTIONS'])
def test_options_dropping_redundant_content_length(httpbin, method):
    if False:
        for i in range(10):
            print('nop')
    r = http('--offline', method, httpbin + '/anything')
    assert 'Content-Length' not in r