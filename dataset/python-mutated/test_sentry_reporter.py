from copy import deepcopy
from unittest.mock import MagicMock, Mock, patch
import pytest
from tribler.core.sentry_reporter.sentry_reporter import BROWSER, CONTEXTS, LAST_CORE_OUTPUT, NAME, OS_ENVIRON, REPORTER, STACKTRACE, SentryReporter, SentryStrategy, TAGS, TRIBLER, TYPE, VALUE, VERSION, this_sentry_strategy
from tribler.core.sentry_reporter.sentry_scrubber import SentryScrubber
from tribler.core.utilities.patch_import import patch_import
DEFAULT_EVENT = {CONTEXTS: {BROWSER: {NAME: TRIBLER, VERSION: '<not set>'}, REPORTER: {}}, TAGS: {}}

@pytest.fixture
def sentry_reporter():
    if False:
        print('Hello World!')
    return SentryReporter()

@patch('tribler.core.sentry_reporter.sentry_reporter.sentry_sdk.init')
def test_init(mocked_init: Mock, sentry_reporter: SentryReporter):
    if False:
        print('Hello World!')
    sentry_reporter.init(sentry_url='url', release_version='release', scrubber=SentryScrubber(), strategy=SentryStrategy.SEND_SUPPRESSED)
    assert sentry_reporter.scrubber
    assert sentry_reporter.global_strategy == SentryStrategy.SEND_SUPPRESSED
    mocked_init.assert_called_once()

@patch('tribler.core.sentry_reporter.sentry_reporter.ignore_logger')
def test_ignore_logger(mocked_ignore_logger: Mock, sentry_reporter: SentryReporter):
    if False:
        return 10
    sentry_reporter.ignore_logger('logger name')
    mocked_ignore_logger.assert_called_with('logger name')

@patch('tribler.core.sentry_reporter.sentry_reporter.sentry_sdk.add_breadcrumb')
def test_add_breadcrumb(mocked_add_breadcrumb: Mock, sentry_reporter: SentryReporter):
    if False:
        print('Hello World!')
    assert sentry_reporter.add_breadcrumb('message', 'category', 'level', named_arg='some')
    mocked_add_breadcrumb.assert_called_with({'message': 'message', 'category': 'category', 'level': 'level'}, named_arg='some')

def test_get_confirmation(sentry_reporter: SentryReporter):
    if False:
        return 10
    mocked_QApplication = Mock()
    mocked_QMessageBox = MagicMock()
    with patch_import('PyQt5.QtWidgets', strict=True, QApplication=mocked_QApplication, QMessageBox=mocked_QMessageBox):
        sentry_reporter.get_confirmation(Exception('test'))
        mocked_QApplication.assert_called()
        mocked_QMessageBox.assert_called()

@patch_import('PyQt5.QtWidgets', always_raise_exception_on_import=True)
def test_get_confirmation_no_qt(sentry_reporter: SentryReporter):
    if False:
        return 10
    assert not sentry_reporter.get_confirmation(Exception('test'))

@patch('tribler.core.sentry_reporter.sentry_reporter.sentry_sdk.capture_exception')
def test_capture_exception(mocked_capture_exception: Mock, sentry_reporter: SentryReporter):
    if False:
        print('Hello World!')
    exception = Exception('test')
    sentry_reporter.capture_exception(exception)
    mocked_capture_exception.assert_called_with(exception)

@patch('tribler.core.sentry_reporter.sentry_reporter.sentry_sdk.capture_exception')
def test_event_from_exception(mocked_capture_exception: Mock, sentry_reporter: SentryReporter):
    if False:
        while True:
            i = 10
    assert sentry_reporter.event_from_exception(None) == {}
    exception = Exception('test')
    sentry_reporter.thread_strategy = Mock()

    def capture_exception(_):
        if False:
            while True:
                i = 10
        sentry_reporter.last_event = {'sentry': 'event'}
    mocked_capture_exception.side_effect = capture_exception
    sentry_reporter.event_from_exception(exception)
    mocked_capture_exception.assert_called_with(exception)
    sentry_reporter.thread_strategy.set.assert_any_call(SentryStrategy.SEND_SUPPRESSED)
    assert sentry_reporter.last_event == {'sentry': 'event'}

def test_set_user(sentry_reporter):
    if False:
        print('Hello World!')
    assert sentry_reporter.set_user(b'some_id') == {'id': 'db69fe66ec6b6b013c2f7d271ce17cae', 'username': 'Wanda Brown'}
    assert sentry_reporter.set_user(b'11111100100') == {'id': '91f900f528d5580581197c2c6a4adbbc', 'username': 'Jennifer Herrera'}

def test_get_actual_strategy(sentry_reporter):
    if False:
        i = 10
        return i + 15
    sentry_reporter.thread_strategy.set(None)
    sentry_reporter.global_strategy = SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION
    assert sentry_reporter.get_actual_strategy() == SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION
    sentry_reporter.thread_strategy.set(SentryStrategy.SEND_ALLOWED)
    assert sentry_reporter.get_actual_strategy() == SentryStrategy.SEND_ALLOWED
    sentry_reporter.thread_strategy.set(None)
    assert sentry_reporter.get_actual_strategy() == SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION

@patch(OS_ENVIRON, {})
def test_get_sentry_url_not_specified():
    if False:
        print('Hello World!')
    assert not SentryReporter.get_sentry_url()

@patch('tribler.core.version.sentry_url', 'sentry_url')
def test_get_sentry_url_from_version_file():
    if False:
        while True:
            i = 10
    assert SentryReporter.get_sentry_url() == 'sentry_url'

@patch(OS_ENVIRON, {'TRIBLER_SENTRY_URL': 'env_url'})
def test_get_sentry_url_from_env():
    if False:
        return 10
    assert SentryReporter.get_sentry_url() == 'env_url'

@patch(OS_ENVIRON, {})
def test_is_not_in_test_mode():
    if False:
        print('Hello World!')
    assert SentryReporter.get_test_sentry_url() is None
    assert not SentryReporter.is_in_test_mode()

@patch(OS_ENVIRON, {'TRIBLER_TEST_SENTRY_URL': 'url'})
def test_is_in_test_mode():
    if False:
        while True:
            i = 10
    assert SentryReporter.get_test_sentry_url() == 'url'
    assert SentryReporter.is_in_test_mode()

def test_before_send_no_event(sentry_reporter: SentryReporter):
    if False:
        while True:
            i = 10
    assert not sentry_reporter._before_send(None, None)

def test_before_send_ignored_exceptions(sentry_reporter: SentryReporter):
    if False:
        while True:
            i = 10
    assert not sentry_reporter._before_send({'some': 'event'}, {'exc_info': [KeyboardInterrupt]})

def test_before_send_suppressed(sentry_reporter: SentryReporter):
    if False:
        i = 10
        return i + 15
    sentry_reporter.global_strategy = SentryStrategy.SEND_SUPPRESSED
    assert not sentry_reporter._before_send({'some': 'event'}, None)
    assert sentry_reporter.last_event == {'some': 'event'}

@patch.object(SentryReporter, 'get_confirmation', lambda _, __: True)
def test_before_send_allowed_with_confiration(sentry_reporter: SentryReporter):
    if False:
        while True:
            i = 10
    sentry_reporter.global_strategy = SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION
    assert sentry_reporter._before_send({'some': 'event'}, None)

def test_before_send_allowed(sentry_reporter: SentryReporter):
    if False:
        while True:
            i = 10
    sentry_reporter.global_strategy = SentryStrategy.SEND_ALLOWED
    assert sentry_reporter._before_send({'some': 'event'}, None)

def test_before_send_scrubber_exists(sentry_reporter: SentryReporter):
    if False:
        i = 10
        return i + 15
    event = {'some': 'event'}
    sentry_reporter.global_strategy = SentryStrategy.SEND_ALLOWED
    sentry_reporter.scrubber = Mock()
    assert sentry_reporter._before_send(event, None)
    sentry_reporter.scrubber.scrub_event.assert_called_with(event)

def test_before_send_scrubber_doesnt_exists(sentry_reporter: SentryReporter):
    if False:
        return 10
    sentry_reporter.scrubber = None
    sentry_reporter.global_strategy = SentryStrategy.SEND_ALLOWED
    assert sentry_reporter._before_send({'some': 'event'}, None)

def test_send_defaults(sentry_reporter):
    if False:
        for i in range(10):
            print('nop')
    assert sentry_reporter.send_event(event={}) == DEFAULT_EVENT

def test_send_additional_tags(sentry_reporter):
    if False:
        return 10
    tags = {'tag_key': 'tag_value', 'numeric_tag_key': 1}
    actual = sentry_reporter.send_event(event={}, tags=tags)
    expected = deepcopy(DEFAULT_EVENT)
    expected[TAGS].update(tags)
    assert actual == expected

def test_before_send(sentry_reporter):
    if False:
        i = 10
        return i + 15
    sentry_reporter.thread_strategy.set(None)
    scrubber = SentryScrubber()
    sentry_reporter.init('', scrubber=scrubber)
    sentry_reporter.last_event = None
    assert sentry_reporter._before_send({}, {}) == {}
    assert sentry_reporter._before_send(None, {}) is None
    assert sentry_reporter._before_send(None, None) is None
    sentry_reporter.global_strategy = SentryStrategy.SEND_SUPPRESSED
    assert sentry_reporter.last_event is None
    assert sentry_reporter._before_send({'a': 'b'}, None) is None
    assert sentry_reporter.last_event == {'a': 'b'}
    sentry_reporter.global_strategy = SentryStrategy.SEND_ALLOWED
    assert sentry_reporter._before_send({'c': 'd'}, None) == {'c': 'd'}
    assert sentry_reporter.last_event == {'a': 'b'}
    assert sentry_reporter._before_send({'a': 'b'}, {'exc_info': [KeyboardInterrupt]}) is None
    assert sentry_reporter._before_send({CONTEXTS: {REPORTER: {STACKTRACE: ['/Users/username/']}}}, None) == {CONTEXTS: {REPORTER: {STACKTRACE: ['/Users/<highlight>/']}}}
    assert sentry_reporter._before_send({'release': '7.6.0'}, None) == {'release': '7.6.0'}
    assert sentry_reporter._before_send({'release': '7.6.0-GIT'}, None) == {'release': 'dev'}
    sentry_reporter.global_strategy = SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION
    sentry_reporter.get_confirmation = lambda e: False
    assert sentry_reporter._before_send({'a': 'b'}, None) is None
    sentry_reporter.get_confirmation = lambda e: True
    assert sentry_reporter._before_send({'a': 'b'}, None) == {'a': 'b'}

def test_sentry_strategy(sentry_reporter):
    if False:
        for i in range(10):
            print('nop')
    sentry_reporter.thread_strategy.set(None)
    sentry_reporter.global_strategy = SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION
    with this_sentry_strategy(sentry_reporter, SentryStrategy.SEND_ALLOWED) as reporter:
        assert reporter.global_strategy == SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION
        assert reporter.thread_strategy.get() == SentryStrategy.SEND_ALLOWED
    assert sentry_reporter.thread_strategy.get() is None
    assert sentry_reporter.global_strategy == SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION

def test_send_last_core_output(sentry_reporter):
    if False:
        for i in range(10):
            print('nop')
    event = {'exception': {'values': [{'module': 'tribler.gui.utilities', TYPE: 'CreationTraceback', VALUE: '\n  File "/Users/<user>/Projects/github.com/Tribler/tribler/src/run_tribler.py", ', 'mechanism': None}, {'module': 'tribler.gui.exceptions', TYPE: 'CoreCrashedError', VALUE: 'The Tribler core has unexpectedly finished with exit code 1 and status: 0.', 'mechanism': None, 'stacktrace': {'frames': []}}]}}
    last_core_output = '\nFile "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.8/lib/python3.8/asyncio/base_events.py", line 1461, in create_server\n    sock.bind(sa)\nOverflowError: bind(): port must be 0-65535.Sentry is attempting to send 1 pending error messages\nWaiting up to 2 seconds\nPress Ctrl-C to quit\n    '
    actual = sentry_reporter.send_event(event=event, last_core_output=last_core_output)
    expected = deepcopy(DEFAULT_EVENT)
    expected['exception'] = {'values': [{'module': 'tribler.gui.exceptions', TYPE: 'CoreCrashedError', VALUE: 'The Tribler core has unexpectedly finished with exit code 1 and status: 0.', 'mechanism': None}, {TYPE: 'OverflowError', VALUE: 'bind(): port must be 0-65535.'}]}
    expected[CONTEXTS][REPORTER][LAST_CORE_OUTPUT] = last_core_output.split('\n')
    assert actual == expected