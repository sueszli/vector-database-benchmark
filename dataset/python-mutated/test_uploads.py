import os
import json
import sys
import subprocess
import time
import contextlib
import httpie.__main__ as main
import pytest
from httpie.cli.exceptions import ParseError
from httpie.client import FORM_CONTENT_TYPE
from httpie.compat import is_windows
from httpie.status import ExitStatus
from .utils import MockEnvironment, StdinBytesIO, http, HTTP_OK
from .fixtures import FILE_PATH_ARG, FILE_PATH, FILE_CONTENT
MAX_RESPONSE_WAIT_TIME = 5

def test_chunked_json(httpbin_with_chunked_support):
    if False:
        for i in range(10):
            print('nop')
    r = http('--verbose', '--chunked', httpbin_with_chunked_support + '/post', 'hello=world')
    assert HTTP_OK in r
    assert 'Transfer-Encoding: chunked' in r
    assert r.count('hello') == 3

def test_chunked_form(httpbin_with_chunked_support):
    if False:
        while True:
            i = 10
    r = http('--verbose', '--chunked', '--form', httpbin_with_chunked_support + '/post', 'hello=world')
    assert HTTP_OK in r
    assert 'Transfer-Encoding: chunked' in r
    assert r.count('hello') == 2

def test_chunked_stdin(httpbin_with_chunked_support):
    if False:
        for i in range(10):
            print('nop')
    r = http('--verbose', '--chunked', httpbin_with_chunked_support + '/post', env=MockEnvironment(stdin=StdinBytesIO(FILE_PATH.read_bytes()), stdin_isatty=False))
    assert HTTP_OK in r
    assert 'Transfer-Encoding: chunked' in r
    assert r.count(FILE_CONTENT) == 2

def test_chunked_stdin_multiple_chunks(httpbin_with_chunked_support):
    if False:
        print('Hello World!')
    data = FILE_PATH.read_bytes()
    stdin_bytes = data + b'\n' + data
    r = http('--verbose', '--chunked', httpbin_with_chunked_support + '/post', env=MockEnvironment(stdin=StdinBytesIO(stdin_bytes), stdin_isatty=False, stdout_isatty=True))
    assert HTTP_OK in r
    assert 'Transfer-Encoding: chunked' in r
    assert r.count(FILE_CONTENT) == 4

def test_chunked_raw(httpbin_with_chunked_support):
    if False:
        i = 10
        return i + 15
    r = http('--verbose', '--chunked', httpbin_with_chunked_support + '/post', '--raw', json.dumps({'a': 1, 'b': '2fafds', 'c': '🥰'}))
    assert HTTP_OK in r
    assert 'Transfer-Encoding: chunked' in r

@contextlib.contextmanager
def stdin_processes(httpbin, *args, warn_threshold=0.1):
    if False:
        return 10
    process_1 = subprocess.Popen(['cat'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    process_2 = subprocess.Popen([sys.executable, main.__file__, 'POST', httpbin + '/post', *args], stdin=process_1.stdout, stderr=subprocess.PIPE, env={**os.environ, 'HTTPIE_STDIN_READ_WARN_THRESHOLD': str(warn_threshold)})
    try:
        yield (process_1, process_2)
    finally:
        process_1.terminate()
        process_2.terminate()

@pytest.mark.parametrize('wait', (True, False))
@pytest.mark.requires_external_processes
@pytest.mark.skipif(is_windows, reason="Windows doesn't support select() calls into files")
def test_reading_from_stdin(httpbin, wait):
    if False:
        print('Hello World!')
    with stdin_processes(httpbin) as (process_1, process_2):
        process_1.communicate(timeout=0.1, input=b'bleh')
        if wait:
            time.sleep(1)
        try:
            (_, errs) = process_2.communicate(timeout=MAX_RESPONSE_WAIT_TIME)
        except subprocess.TimeoutExpired:
            errs = b''
        assert b'> warning: no stdin data read in 0.1s' not in errs

@pytest.mark.requires_external_processes
@pytest.mark.skipif(is_windows, reason="Windows doesn't support select() calls into files")
def test_stdin_read_warning(httpbin):
    if False:
        return 10
    with stdin_processes(httpbin) as (process_1, process_2):
        time.sleep(1)
        process_1.communicate(timeout=0.1, input=b'bleh\n')
        try:
            (_, errs) = process_2.communicate(timeout=MAX_RESPONSE_WAIT_TIME)
        except subprocess.TimeoutExpired:
            errs = b''
        assert b'> warning: no stdin data read in 0.1s' in errs

@pytest.mark.requires_external_processes
@pytest.mark.skipif(is_windows, reason="Windows doesn't support select() calls into files")
def test_stdin_read_warning_with_quiet(httpbin):
    if False:
        i = 10
        return i + 15
    with stdin_processes(httpbin, '-qq') as (process_1, process_2):
        time.sleep(1)
        process_1.communicate(timeout=0.1, input=b'bleh\n')
        try:
            (_, errs) = process_2.communicate(timeout=MAX_RESPONSE_WAIT_TIME)
        except subprocess.TimeoutExpired:
            errs = b''
        assert b'> warning: no stdin data read in 0.1s' not in errs

@pytest.mark.requires_external_processes
@pytest.mark.skipif(is_windows, reason="Windows doesn't support select() calls into files")
def test_stdin_read_warning_blocking_exit(httpbin):
    if False:
        for i in range(10):
            print('nop')
    with stdin_processes(httpbin, warn_threshold=999) as (process_1, process_2):
        time.sleep(1)
        process_1.communicate(timeout=0.1, input=b'some input\n')
        process_2.communicate(timeout=MAX_RESPONSE_WAIT_TIME)

class TestMultipartFormDataFileUpload:

    def test_non_existent_file_raises_parse_error(self, httpbin):
        if False:
            while True:
                i = 10
        with pytest.raises(ParseError):
            http('--form', 'POST', httpbin.url + '/post', 'foo@/__does_not_exist__')

    def test_upload_ok(self, httpbin):
        if False:
            for i in range(10):
                print('nop')
        r = http('--form', '--verbose', 'POST', httpbin.url + '/post', f'test-file@{FILE_PATH_ARG}', 'foo=bar')
        assert HTTP_OK in r
        assert 'Content-Disposition: form-data; name="foo"' in r
        assert f'Content-Disposition: form-data; name="test-file"; filename="{os.path.basename(FILE_PATH)}"' in r
        assert FILE_CONTENT in r
        assert '"foo": "bar"' in r
        assert 'Content-Type: text/plain' in r

    def test_upload_multiple_fields_with_the_same_name(self, httpbin):
        if False:
            print('Hello World!')
        r = http('--form', '--verbose', 'POST', httpbin.url + '/post', f'test-file@{FILE_PATH_ARG}', f'test-file@{FILE_PATH_ARG}')
        assert HTTP_OK in r
        assert r.count(f'Content-Disposition: form-data; name="test-file"; filename="{os.path.basename(FILE_PATH)}"') == 2
        assert r.count(FILE_CONTENT) in [3, 4]
        assert r.count('Content-Type: text/plain') == 2

    def test_upload_custom_content_type(self, httpbin):
        if False:
            i = 10
            return i + 15
        r = http('--form', '--verbose', httpbin.url + '/post', f'test-file@{FILE_PATH_ARG};type=image/vnd.microsoft.icon')
        assert HTTP_OK in r
        assert f'Content-Disposition: form-data; name="test-file"; filename="{os.path.basename(FILE_PATH)}"' in r
        assert r.count(FILE_CONTENT) == 2
        assert 'Content-Type: image/vnd.microsoft.icon' in r

    def test_form_no_files_urlencoded(self, httpbin):
        if False:
            for i in range(10):
                print('nop')
        r = http('--form', '--verbose', httpbin.url + '/post', 'AAAA=AAA', 'BBB=BBB')
        assert HTTP_OK in r
        assert FORM_CONTENT_TYPE in r

    def test_multipart(self, httpbin):
        if False:
            while True:
                i = 10
        r = http('--verbose', '--multipart', httpbin.url + '/post', 'AAAA=AAA', 'BBB=BBB')
        assert HTTP_OK in r
        assert FORM_CONTENT_TYPE not in r
        assert 'multipart/form-data' in r

    def test_form_multipart_custom_boundary(self, httpbin):
        if False:
            while True:
                i = 10
        boundary = 'HTTPIE_FTW'
        r = http('--print=HB', '--check-status', '--multipart', f'--boundary={boundary}', httpbin.url + '/post', 'AAAA=AAA', 'BBB=BBB')
        assert f'multipart/form-data; boundary={boundary}' in r
        assert r.count(boundary) == 4

    def test_multipart_custom_content_type_boundary_added(self, httpbin):
        if False:
            print('Hello World!')
        boundary = 'HTTPIE_FTW'
        r = http('--print=HB', '--check-status', '--multipart', f'--boundary={boundary}', httpbin.url + '/post', 'Content-Type: multipart/magic', 'AAAA=AAA', 'BBB=BBB')
        assert f'multipart/magic; boundary={boundary}' in r
        assert r.count(boundary) == 4

    def test_multipart_custom_content_type_boundary_preserved(self, httpbin):
        if False:
            i = 10
            return i + 15
        boundary_in_header = 'HEADER_BOUNDARY'
        boundary_in_body = 'BODY_BOUNDARY'
        r = http('--print=HB', '--check-status', '--multipart', f'--boundary={boundary_in_body}', httpbin.url + '/post', f'Content-Type: multipart/magic; boundary={boundary_in_header}', 'AAAA=AAA', 'BBB=BBB')
        assert f'multipart/magic; boundary={boundary_in_header}' in r
        assert r.count(boundary_in_body) == 3

    def test_multipart_chunked(self, httpbin_with_chunked_support):
        if False:
            return 10
        r = http('--verbose', '--multipart', '--chunked', httpbin_with_chunked_support + '/post', 'AAA=AAA')
        assert 'Transfer-Encoding: chunked' in r
        assert 'multipart/form-data' in r
        assert 'name="AAA"' in r
        assert '"AAA": "AAA"', r

    def test_multipart_preserve_order(self, httpbin):
        if False:
            while True:
                i = 10
        r = http('--form', '--offline', httpbin + '/post', 'text_field=foo', f'file_field@{FILE_PATH_ARG}')
        assert r.index('text_field') < r.index('file_field')
        r = http('--form', '--offline', httpbin + '/post', f'file_field@{FILE_PATH_ARG}', 'text_field=foo')
        assert r.index('text_field') > r.index('file_field')

class TestRequestBodyFromFilePath:
    """
    `http URL @file'

    """

    def test_request_body_from_file_by_path(self, httpbin):
        if False:
            while True:
                i = 10
        r = http('--verbose', 'POST', httpbin.url + '/post', '@' + FILE_PATH_ARG)
        assert HTTP_OK in r
        assert r.count(FILE_CONTENT) == 2
        assert '"Content-Type": "text/plain"' in r

    def test_request_body_from_file_by_path_chunked(self, httpbin_with_chunked_support):
        if False:
            i = 10
            return i + 15
        r = http('--verbose', '--chunked', httpbin_with_chunked_support + '/post', '@' + FILE_PATH_ARG)
        assert HTTP_OK in r
        assert 'Transfer-Encoding: chunked' in r
        assert '"Content-Type": "text/plain"' in r
        assert r.count(FILE_CONTENT) == 2

    def test_request_body_from_file_by_path_with_explicit_content_type(self, httpbin):
        if False:
            while True:
                i = 10
        r = http('--verbose', 'POST', httpbin.url + '/post', '@' + FILE_PATH_ARG, 'Content-Type:text/plain; charset=UTF-8')
        assert HTTP_OK in r
        assert FILE_CONTENT in r
        assert 'Content-Type: text/plain; charset=UTF-8' in r

    def test_request_body_from_file_by_path_no_field_name_allowed(self, httpbin):
        if False:
            print('Hello World!')
        env = MockEnvironment(stdin_isatty=True)
        r = http('POST', httpbin.url + '/post', 'field-name@' + FILE_PATH_ARG, env=env, tolerate_error_exit_status=True)
        assert 'perhaps you meant --form?' in r.stderr

    def test_request_body_from_file_by_path_no_data_items_allowed(self, httpbin):
        if False:
            print('Hello World!')
        env = MockEnvironment(stdin_isatty=False)
        r = http('POST', httpbin.url + '/post', '@' + FILE_PATH_ARG, 'foo=bar', env=env, tolerate_error_exit_status=True)
        assert r.exit_status == ExitStatus.ERROR
        assert 'cannot be mixed' in r.stderr

    def test_multiple_request_bodies_from_file_by_path(self, httpbin):
        if False:
            print('Hello World!')
        env = MockEnvironment(stdin_isatty=True)
        r = http('--verbose', 'POST', httpbin.url + '/post', '@' + FILE_PATH_ARG, '@' + FILE_PATH_ARG, env=env, tolerate_error_exit_status=True)
        assert r.exit_status == ExitStatus.ERROR
        assert 'from multiple files' in r.stderr