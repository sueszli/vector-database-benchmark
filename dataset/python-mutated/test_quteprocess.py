"""Test the quteproc fixture used for tests."""
import logging
import datetime
import json
import pytest
from end2end.fixtures import quteprocess, testprocess
from qutebrowser.utils import log

class FakeRepCall:
    """Fake for request.node.rep_call."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.failed = False
        self.skipped = False

class FakeConfig:
    """Fake for request.config."""
    ARGS = {'--qute-delay': 0, '--color': True, '--verbose': False, '--capture': None}
    INI = {'qt_log_ignore': []}

    def __init__(self):
        if False:
            while True:
                i = 10
        self.webengine = False

    def getoption(self, name):
        if False:
            i = 10
            return i + 15
        return self.ARGS[name]

    def getini(self, name):
        if False:
            print('Hello World!')
        return self.INI[name]

class FakeNode:
    """Fake for request.node."""

    def __init__(self, call):
        if False:
            i = 10
            return i + 15
        self.rep_call = call

    def get_closest_marker(self, _name):
        if False:
            i = 10
            return i + 15
        return None

class FakeRequest:
    """Fake for request."""

    def __init__(self, node, config, server):
        if False:
            return 10
        self.node = node
        self.config = config
        self._server = server

    def getfixturevalue(self, name):
        if False:
            i = 10
            return i + 15
        assert name == 'server'
        return self._server

@pytest.fixture
def request_mock(quteproc, monkeypatch, server):
    if False:
        i = 10
        return i + 15
    'Patch out a pytest request.'
    fake_call = FakeRepCall()
    fake_config = FakeConfig()
    fake_node = FakeNode(fake_call)
    fake_request = FakeRequest(fake_node, fake_config, server)
    assert not hasattr(fake_request.node.rep_call, 'wasxfail')
    monkeypatch.setattr(quteproc, 'request', fake_request)
    return fake_request

@pytest.mark.parametrize('cmd', [':message-error test', ':jseval console.log("[FAIL] test");'])
def test_quteproc_error_message(qtbot, quteproc, cmd, request_mock):
    if False:
        i = 10
        return i + 15
    'Make sure the test fails with an unexpected error message.'
    with qtbot.wait_signal(quteproc.got_error):
        quteproc.send_cmd(cmd)
    with pytest.raises(pytest.fail.Exception):
        quteproc.after_test()

def test_quteproc_error_message_did_fail(qtbot, quteproc, request_mock):
    if False:
        print('Hello World!')
    'Make sure the test does not fail on teardown if the main test failed.'
    request_mock.node.rep_call.failed = True
    with qtbot.wait_signal(quteproc.got_error):
        quteproc.send_cmd(':message-error test')
    quteproc.after_test()

def test_quteproc_skip_via_js(qtbot, quteproc):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(pytest.skip.Exception, match='test'):
        quteproc.send_cmd(':jseval console.log("[SKIP] test");')
        quteproc.wait_for_js('[SKIP] test')
        quteproc.after_test()

def test_quteproc_skip_and_wait_for(qtbot, quteproc):
    if False:
        print('Hello World!')
    "This test will skip *again* during teardown, but we don't care."
    with pytest.raises(pytest.skip.Exception):
        quteproc.send_cmd(':jseval console.log("[SKIP] foo");')
        quteproc.wait_for_js('[SKIP] foo')
        quteproc.wait_for(message='This will not match')

def test_qt_log_ignore(qtbot, quteproc):
    if False:
        return 10
    'Make sure the test passes when logging a qt_log_ignore message.'
    with qtbot.wait_signal(quteproc.got_error):
        quteproc.send_cmd(':message-error "QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to blabla"')

def test_quteprocess_quitting(qtbot, quteproc_process):
    if False:
        while True:
            i = 10
    'When qutebrowser quits, after_test should fail.'
    with qtbot.wait_signal(quteproc_process.proc.finished, timeout=15000):
        quteproc_process.send_cmd(':quit')
    with pytest.raises(testprocess.ProcessExited):
        quteproc_process.after_test()

@pytest.mark.parametrize('data, attrs', [pytest.param('{"created": 86400, "msecs": 0, "levelname": "DEBUG", "name": "init", "module": "earlyinit", "funcName": "init_log", "lineno": 280, "levelno": 10, "message": "Log initialized."}', {'timestamp': datetime.datetime.fromtimestamp(86400), 'loglevel': logging.DEBUG, 'category': 'init', 'module': 'earlyinit', 'function': 'init_log', 'line': 280, 'message': 'Log initialized.', 'expected': False}, id='normal'), pytest.param('{"created": 86400, "msecs": 0, "levelname": "VDEBUG", "name": "foo", "module": "foo", "funcName": "foo", "lineno": 0, "levelno": 9, "message": ""}', {'loglevel': log.VDEBUG_LEVEL}, id='vdebug'), pytest.param('{"created": 86400, "msecs": 0, "levelname": "DEBUG", "name": "qt", "module": null, "funcName": null, "lineno": 0, "levelno": 10, "message": "test"}', {'module': None, 'function': None, 'line': None}, id='unknown module'), pytest.param('{"created": 86400, "msecs": 0, "levelname": "VDEBUG", "name": "foo", "module": "foo", "funcName": "foo", "lineno": 0, "levelno": 9, "message": "QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to blabla"}', {'expected': True}, id='expected message'), pytest.param('{"created": 86400, "msecs": 0, "levelname": "DEBUG", "name": "qt", "module": "qnetworkreplyhttpimpl", "funcName": "void QNetworkReplyHttpImplPrivate::error(QNetworkReply::NetworkError, const QString&)", "lineno": 1929, "levelno": 10, "message": "QNetworkReplyImplPrivate::error: Internal problem, this method must only be called once."}', {'module': 'qnetworkreplyhttpimpl', 'function': 'void QNetworkReplyHttpImplPrivate::error(QNetworkReply::NetworkError, const QString&)', 'line': 1929}, id='weird Qt location'), pytest.param('{"created": 86400, "msecs": 0, "levelname": "DEBUG", "name": "qt", "module": "qxcbxsettings", "funcName": "QXcbXSettings::QXcbXSettings(QXcbScreen*)", "lineno": 233, "levelno": 10, "message": "QXcbXSettings::QXcbXSettings(QXcbScreen*) Failed to get selection owner for XSETTINGS_S atom"}', {'module': 'qxcbxsettings', 'function': 'QXcbXSettings::QXcbXSettings(QXcbScreen*)', 'line': 233}, id='QXcbXSettings'), pytest.param('{"created": 86400, "msecs": 0, "levelname": "WARNING", "name": "py.warnings", "module": "app", "funcName": "qt_mainloop", "lineno": 121, "levelno": 30, "message": ".../app.py:121: ResourceWarning: unclosed file <_io.TextIOWrapper name=18 mode=\'r\' encoding=\'UTF-8\'>"}', {'category': 'py.warnings'}, id='resourcewarning')])
def test_log_line_parse(pytestconfig, data, attrs):
    if False:
        print('Hello World!')
    line = quteprocess.LogLine(pytestconfig, data)
    for (name, expected) in attrs.items():
        actual = getattr(line, name)
        assert actual == expected, name

@pytest.mark.parametrize('data, colorized, expect_error, expected', [pytest.param({'created': 86400, 'msecs': 0, 'levelname': 'DEBUG', 'name': 'foo', 'module': 'bar', 'funcName': 'qux', 'lineno': 10, 'levelno': 10, 'message': 'quux'}, False, False, '{timestamp} DEBUG    foo        bar:qux:10 quux', id='normal'), pytest.param({'created': 86400, 'msecs': 0, 'levelname': 'DEBUG', 'name': 'foo', 'module': 'bar', 'funcName': 'qux', 'lineno': 10, 'levelno': 10, 'message': 'quux', 'traceback': 'Traceback (most recent call last):\n here be dragons'}, False, False, '{timestamp} DEBUG    foo        bar:qux:10 quux\nTraceback (most recent call last):\n here be dragons', id='traceback'), pytest.param({'created': 86400, 'msecs': 0, 'levelname': 'DEBUG', 'name': 'foo', 'module': 'bar', 'funcName': 'qux', 'lineno': 10, 'levelno': 10, 'message': 'quux'}, True, False, '\x1b[32m{timestamp}\x1b[0m \x1b[37mDEBUG   \x1b[0m \x1b[36mfoo        bar:qux:10\x1b[0m \x1b[37mquux\x1b[0m', id='colored'), pytest.param({'created': 86400, 'msecs': 0, 'levelname': 'ERROR', 'name': 'foo', 'module': 'bar', 'funcName': 'qux', 'lineno': 10, 'levelno': 40, 'message': 'quux'}, False, True, '{timestamp} ERROR (expected) foo        bar:qux:10 quux', id='expected error'), pytest.param({'created': 86400, 'msecs': 0, 'levelname': 'DEBUG', 'name': 'foo', 'module': 'bar', 'funcName': 'qux', 'lineno': 10, 'levelno': 10, 'message': 'quux'}, False, True, '{timestamp} DEBUG    foo        bar:qux:10 quux', id='expected other'), pytest.param({'created': 86400, 'msecs': 0, 'levelname': 'ERROR', 'name': 'foo', 'module': 'bar', 'funcName': 'qux', 'lineno': 10, 'levelno': 40, 'message': 'quux'}, True, True, '\x1b[32m{timestamp}\x1b[0m \x1b[37mERROR (expected)\x1b[0m \x1b[36mfoo        bar:qux:10\x1b[0m \x1b[37mquux\x1b[0m', id='expected error colorized')])
def test_log_line_formatted(pytestconfig, data, colorized, expect_error, expected):
    if False:
        i = 10
        return i + 15
    line = json.dumps(data)
    record = quteprocess.LogLine(pytestconfig, line)
    record.expected = expect_error
    ts = datetime.datetime.fromtimestamp(data['created']).strftime('%H:%M:%S')
    ts += '.{:03.0f}'.format(data['msecs'])
    expected = expected.format(timestamp=ts)
    assert record.formatted_str(colorized=colorized) == expected

def test_log_line_no_match(pytestconfig):
    if False:
        i = 10
        return i + 15
    with pytest.raises(testprocess.InvalidLine):
        quteprocess.LogLine(pytestconfig, 'Hello World!')

class TestClickElementByText:

    @pytest.fixture(autouse=True)
    def open_page(self, quteproc):
        if False:
            for i in range(10):
                print('nop')
        quteproc.open_path('data/click_element.html')

    def test_click_element(self, quteproc):
        if False:
            for i in range(10):
                print('nop')
        quteproc.click_element_by_text('Test Element')
        quteproc.wait_for_js('click_element clicked')

    def test_click_special_chars(self, quteproc):
        if False:
            while True:
                i = 10
        quteproc.click_element_by_text('"Don\'t", he shouted')
        quteproc.wait_for_js('click_element special chars')

    def test_duplicate(self, quteproc):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError, match='not unique'):
            quteproc.click_element_by_text('Duplicate')

    def test_nonexistent(self, quteproc):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError, match='No element'):
            quteproc.click_element_by_text('no element exists with this text')

@pytest.mark.parametrize('string, expected', [('Test', "'Test'"), ("Don't", '"Don\'t"'), ('"Don\'t", he said', 'concat(\'"\', \'Don\', "\'", \'t\', \'"\', \', he said\')')])
def test_xpath_escape(string, expected):
    if False:
        for i in range(10):
            print('nop')
    assert quteprocess._xpath_escape(string) == expected

@pytest.mark.parametrize('value', ['foo', 'foo"bar'])
def test_set(quteproc, value):
    if False:
        i = 10
        return i + 15
    quteproc.set_setting('content.default_encoding', value)
    read_back = quteproc.get_setting('content.default_encoding')
    assert read_back == value

@pytest.mark.parametrize('message, ignored', [('Hello World', False), ('[0509/185222.442798:ERROR:ssl_client_socket_impl.cc(959)] handshake failed; returned -1, SSL error code 1, net_error -202', True), ('[503633:503650:0509/185222.442798:ERROR:ssl_client_socket_impl.cc(959)] handshake failed; returned -1, SSL error code 1, net_error -202', True), ('[26598:26598:0605/191429.639416:WARNING:audio_manager.cc(317)] Test', False)])
def test_is_ignored_chromium_message(message, ignored):
    if False:
        return 10
    assert quteprocess.is_ignored_chromium_message(message) == ignored