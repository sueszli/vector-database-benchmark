from unittest import mock
from httpie.status import ExitStatus
from .utils import MockEnvironment, http, HTTP_OK

def test_keyboard_interrupt_during_arg_parsing_exit_status(httpbin):
    if False:
        return 10
    with mock.patch('httpie.cli.definition.parser.parse_args', side_effect=KeyboardInterrupt()):
        r = http('GET', httpbin.url + '/get', tolerate_error_exit_status=True)
        assert r.exit_status == ExitStatus.ERROR_CTRL_C

def test_keyboard_interrupt_in_program_exit_status(httpbin):
    if False:
        print('Hello World!')
    with mock.patch('httpie.core.program', side_effect=KeyboardInterrupt()):
        r = http('GET', httpbin.url + '/get', tolerate_error_exit_status=True)
        assert r.exit_status == ExitStatus.ERROR_CTRL_C

def test_ok_response_exits_0(httpbin):
    if False:
        print('Hello World!')
    r = http('GET', httpbin.url + '/get')
    assert HTTP_OK in r
    assert r.exit_status == ExitStatus.SUCCESS

def test_error_response_exits_0_without_check_status(httpbin):
    if False:
        for i in range(10):
            print('nop')
    r = http('GET', httpbin.url + '/status/500')
    assert '500 INTERNAL SERVER ERROR' in r
    assert r.exit_status == ExitStatus.SUCCESS
    assert not r.stderr

def test_timeout_exit_status(httpbin):
    if False:
        return 10
    r = http('--timeout=0.01', 'GET', httpbin.url + '/delay/0.5', tolerate_error_exit_status=True)
    assert r.exit_status == ExitStatus.ERROR_TIMEOUT

def test_3xx_check_status_exits_3_and_stderr_when_stdout_redirected(httpbin):
    if False:
        while True:
            i = 10
    env = MockEnvironment(stdout_isatty=False)
    r = http('--check-status', '--headers', 'GET', httpbin.url + '/status/301', env=env, tolerate_error_exit_status=True)
    assert '301 MOVED PERMANENTLY' in r
    assert r.exit_status == ExitStatus.ERROR_HTTP_3XX
    assert '301 moved permanently' in r.stderr.lower()

def test_3xx_check_status_redirects_allowed_exits_0(httpbin):
    if False:
        i = 10
        return i + 15
    r = http('--check-status', '--follow', 'GET', httpbin.url + '/status/301', tolerate_error_exit_status=True)
    assert HTTP_OK in r
    assert r.exit_status == ExitStatus.SUCCESS

def test_4xx_check_status_exits_4(httpbin):
    if False:
        for i in range(10):
            print('nop')
    r = http('--check-status', 'GET', httpbin.url + '/status/401', tolerate_error_exit_status=True)
    assert '401 UNAUTHORIZED' in r
    assert r.exit_status == ExitStatus.ERROR_HTTP_4XX
    assert not r.stderr

def test_5xx_check_status_exits_5(httpbin):
    if False:
        print('Hello World!')
    r = http('--check-status', 'GET', httpbin.url + '/status/500', tolerate_error_exit_status=True)
    assert '500 INTERNAL SERVER ERROR' in r
    assert r.exit_status == ExitStatus.ERROR_HTTP_5XX