import logging
from typing import Iterator
import pytest
from _pytest.logging import caplog_records_key
from _pytest.pytester import Pytester
logger = logging.getLogger(__name__)
sublogger = logging.getLogger(__name__ + '.baz')

@pytest.fixture(autouse=True)
def cleanup_disabled_logging() -> Iterator[None]:
    if False:
        while True:
            i = 10
    "Simple fixture that ensures that a test doesn't disable logging.\n\n    This is necessary because ``logging.disable()`` is global, so a test disabling logging\n    and not cleaning up after will break every test that runs after it.\n\n    This behavior was moved to a fixture so that logging will be un-disabled even if the test fails an assertion.\n    "
    yield
    logging.disable(logging.NOTSET)

def test_fixture_help(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    result = pytester.runpytest('--fixtures')
    result.stdout.fnmatch_lines(['*caplog*'])

def test_change_level(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        while True:
            i = 10
    caplog.set_level(logging.INFO)
    logger.debug('handler DEBUG level')
    logger.info('handler INFO level')
    caplog.set_level(logging.CRITICAL, logger=sublogger.name)
    sublogger.warning('logger WARNING level')
    sublogger.critical('logger CRITICAL level')
    assert 'DEBUG' not in caplog.text
    assert 'INFO' in caplog.text
    assert 'WARNING' not in caplog.text
    assert 'CRITICAL' in caplog.text

def test_change_level_logging_disabled(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        while True:
            i = 10
    logging.disable(logging.CRITICAL)
    assert logging.root.manager.disable == logging.CRITICAL
    caplog.set_level(logging.WARNING)
    logger.info('handler INFO level')
    logger.warning('handler WARNING level')
    caplog.set_level(logging.CRITICAL, logger=sublogger.name)
    sublogger.warning('logger SUB_WARNING level')
    sublogger.critical('logger SUB_CRITICAL level')
    assert 'INFO' not in caplog.text
    assert 'WARNING' in caplog.text
    assert 'SUB_WARNING' not in caplog.text
    assert 'SUB_CRITICAL' in caplog.text

def test_change_level_undo(pytester: Pytester) -> None:
    if False:
        return 10
    "Ensure that 'set_level' is undone after the end of the test.\n\n    Tests the logging output themselves (affected both by logger and handler levels).\n    "
    pytester.makepyfile("\n        import logging\n\n        def test1(caplog):\n            caplog.set_level(logging.INFO)\n            # using + operator here so fnmatch_lines doesn't match the code in the traceback\n            logging.info('log from ' + 'test1')\n            assert 0\n\n        def test2(caplog):\n            # using + operator here so fnmatch_lines doesn't match the code in the traceback\n            logging.info('log from ' + 'test2')\n            assert 0\n    ")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*log from test1*', '*2 failed in *'])
    result.stdout.no_fnmatch_line('*log from test2*')

def test_change_disabled_level_undo(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Ensure that '_force_enable_logging' in 'set_level' is undone after the end of the test.\n\n    Tests the logging output themselves (affected by disabled logging level).\n    "
    pytester.makepyfile("\n        import logging\n\n        def test1(caplog):\n            logging.disable(logging.CRITICAL)\n            caplog.set_level(logging.INFO)\n            # using + operator here so fnmatch_lines doesn't match the code in the traceback\n            logging.info('log from ' + 'test1')\n            assert 0\n\n        def test2(caplog):\n            # using + operator here so fnmatch_lines doesn't match the code in the traceback\n            # use logging.warning because we need a level that will show up if logging.disabled\n            # isn't reset to ``CRITICAL`` after test1.\n            logging.warning('log from ' + 'test2')\n            assert 0\n    ")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*log from test1*', '*2 failed in *'])
    result.stdout.no_fnmatch_line('*log from test2*')

def test_change_level_undos_handler_level(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Ensure that 'set_level' is undone after the end of the test (handler).\n\n    Issue #7569. Tests the handler level specifically.\n    "
    pytester.makepyfile('\n        import logging\n\n        def test1(caplog):\n            assert caplog.handler.level == 0\n            caplog.set_level(9999)\n            caplog.set_level(41)\n            assert caplog.handler.level == 41\n\n        def test2(caplog):\n            assert caplog.handler.level == 0\n\n        def test3(caplog):\n            assert caplog.handler.level == 0\n            caplog.set_level(43)\n            assert caplog.handler.level == 43\n    ')
    result = pytester.runpytest()
    result.assert_outcomes(passed=3)

def test_with_statement(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        return 10
    with caplog.at_level(logging.INFO):
        logger.debug('handler DEBUG level')
        logger.info('handler INFO level')
        with caplog.at_level(logging.CRITICAL, logger=sublogger.name):
            sublogger.warning('logger WARNING level')
            sublogger.critical('logger CRITICAL level')
    assert 'DEBUG' not in caplog.text
    assert 'INFO' in caplog.text
    assert 'WARNING' not in caplog.text
    assert 'CRITICAL' in caplog.text

def test_with_statement_logging_disabled(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    logging.disable(logging.CRITICAL)
    assert logging.root.manager.disable == logging.CRITICAL
    with caplog.at_level(logging.WARNING):
        logger.debug('handler DEBUG level')
        logger.info('handler INFO level')
        logger.warning('handler WARNING level')
        logger.error('handler ERROR level')
        logger.critical('handler CRITICAL level')
        assert logging.root.manager.disable == logging.INFO
        with caplog.at_level(logging.CRITICAL, logger=sublogger.name):
            sublogger.warning('logger SUB_WARNING level')
            sublogger.critical('logger SUB_CRITICAL level')
    assert 'DEBUG' not in caplog.text
    assert 'INFO' not in caplog.text
    assert 'WARNING' in caplog.text
    assert 'ERROR' in caplog.text
    assert ' CRITICAL' in caplog.text
    assert 'SUB_WARNING' not in caplog.text
    assert 'SUB_CRITICAL' in caplog.text
    assert logging.root.manager.disable == logging.CRITICAL

@pytest.mark.parametrize('level_str,expected_disable_level', [('CRITICAL', logging.ERROR), ('ERROR', logging.WARNING), ('WARNING', logging.INFO), ('INFO', logging.DEBUG), ('DEBUG', logging.NOTSET), ('NOTSET', logging.NOTSET), ('NOTVALIDLEVEL', logging.NOTSET)])
def test_force_enable_logging_level_string(caplog: pytest.LogCaptureFixture, level_str: str, expected_disable_level: int) -> None:
    if False:
        while True:
            i = 10
    'Test _force_enable_logging using a level string.\n\n    ``expected_disable_level`` is one level below ``level_str`` because the disabled log level\n    always needs to be *at least* one level lower than the level that caplog is trying to capture.\n    '
    test_logger = logging.getLogger('test_str_level_force_enable')
    logging.disable(logging.CRITICAL)
    assert not test_logger.isEnabledFor(logging.CRITICAL)
    caplog._force_enable_logging(level_str, test_logger)
    assert test_logger.manager.disable == expected_disable_level

def test_log_access(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    caplog.set_level(logging.INFO)
    logger.info('boo %s', 'arg')
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].msg == 'boo %s'
    assert 'boo arg' in caplog.text

def test_messages(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    caplog.set_level(logging.INFO)
    logger.info('boo %s', 'arg')
    logger.info('bar %s\nbaz %s', 'arg1', 'arg2')
    assert 'boo arg' == caplog.messages[0]
    assert 'bar arg1\nbaz arg2' == caplog.messages[1]
    assert caplog.text.count('\n') > len(caplog.messages)
    assert len(caplog.text.splitlines()) > len(caplog.messages)
    try:
        raise Exception('test')
    except Exception:
        logger.exception('oops')
    assert 'oops' in caplog.text
    assert 'oops' in caplog.messages[-1]
    assert 'Exception' in caplog.text
    assert 'Exception' not in caplog.messages[-1]

def test_record_tuples(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        print('Hello World!')
    caplog.set_level(logging.INFO)
    logger.info('boo %s', 'arg')
    assert caplog.record_tuples == [(__name__, logging.INFO, 'boo arg')]

def test_unicode(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    caplog.set_level(logging.INFO)
    logger.info('bū')
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].msg == 'bū'
    assert 'bū' in caplog.text

def test_clear(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        print('Hello World!')
    caplog.set_level(logging.INFO)
    logger.info('bū')
    assert len(caplog.records)
    assert caplog.text
    caplog.clear()
    assert not len(caplog.records)
    assert not caplog.text

@pytest.fixture
def logging_during_setup_and_teardown(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    if False:
        for i in range(10):
            print('nop')
    caplog.set_level('INFO')
    logger.info('a_setup_log')
    yield
    logger.info('a_teardown_log')
    assert [x.message for x in caplog.get_records('teardown')] == ['a_teardown_log']

def test_caplog_captures_for_all_stages(caplog: pytest.LogCaptureFixture, logging_during_setup_and_teardown: None) -> None:
    if False:
        print('Hello World!')
    assert not caplog.records
    assert not caplog.get_records('call')
    logger.info('a_call_log')
    assert [x.message for x in caplog.get_records('call')] == ['a_call_log']
    assert [x.message for x in caplog.get_records('setup')] == ['a_setup_log']
    caplog_records = caplog._item.stash[caplog_records_key]
    assert set(caplog_records) == {'setup', 'call'}

def test_clear_for_call_stage(caplog: pytest.LogCaptureFixture, logging_during_setup_and_teardown: None) -> None:
    if False:
        print('Hello World!')
    logger.info('a_call_log')
    assert [x.message for x in caplog.get_records('call')] == ['a_call_log']
    assert [x.message for x in caplog.get_records('setup')] == ['a_setup_log']
    caplog_records = caplog._item.stash[caplog_records_key]
    assert set(caplog_records) == {'setup', 'call'}
    caplog.clear()
    assert caplog.get_records('call') == []
    assert [x.message for x in caplog.get_records('setup')] == ['a_setup_log']
    caplog_records = caplog._item.stash[caplog_records_key]
    assert set(caplog_records) == {'setup', 'call'}
    logging.info('a_call_log_after_clear')
    assert [x.message for x in caplog.get_records('call')] == ['a_call_log_after_clear']
    assert [x.message for x in caplog.get_records('setup')] == ['a_setup_log']
    caplog_records = caplog._item.stash[caplog_records_key]
    assert set(caplog_records) == {'setup', 'call'}

def test_ini_controls_global_log_level(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile('\n        import pytest\n        import logging\n        def test_log_level_override(request, caplog):\n            plugin = request.config.pluginmanager.getplugin(\'logging-plugin\')\n            assert plugin.log_level == logging.ERROR\n            logger = logging.getLogger(\'catchlog\')\n            logger.warning("WARNING message won\'t be shown")\n            logger.error("ERROR message will be shown")\n            assert \'WARNING\' not in caplog.text\n            assert \'ERROR\' in caplog.text\n    ')
    pytester.makeini('\n        [pytest]\n        log_level=ERROR\n    ')
    result = pytester.runpytest()
    assert result.ret == 0

def test_caplog_can_override_global_log_level(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        import pytest\n        import logging\n        def test_log_level_override(request, caplog):\n            logger = logging.getLogger(\'catchlog\')\n            plugin = request.config.pluginmanager.getplugin(\'logging-plugin\')\n            assert plugin.log_level == logging.WARNING\n\n            logger.info("INFO message won\'t be shown")\n\n            caplog.set_level(logging.INFO, logger.name)\n\n            with caplog.at_level(logging.DEBUG, logger.name):\n                logger.debug("DEBUG message will be shown")\n\n            logger.debug("DEBUG message won\'t be shown")\n\n            with caplog.at_level(logging.CRITICAL, logger.name):\n                logger.warning("WARNING message won\'t be shown")\n\n            logger.debug("DEBUG message won\'t be shown")\n            logger.info("INFO message will be shown")\n\n            assert "message won\'t be shown" not in caplog.text\n    ')
    pytester.makeini('\n        [pytest]\n        log_level=WARNING\n    ')
    result = pytester.runpytest()
    assert result.ret == 0

def test_caplog_captures_despite_exception(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        import pytest\n        import logging\n        def test_log_level_override(request, caplog):\n            logger = logging.getLogger(\'catchlog\')\n            plugin = request.config.pluginmanager.getplugin(\'logging-plugin\')\n            assert plugin.log_level == logging.WARNING\n\n            logger.error("ERROR message " + "will be shown")\n\n            with caplog.at_level(logging.DEBUG, logger.name):\n                logger.debug("DEBUG message " + "won\'t be shown")\n                raise Exception()\n    ')
    pytester.makeini('\n        [pytest]\n        log_level=WARNING\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*ERROR message will be shown*'])
    result.stdout.no_fnmatch_line("*DEBUG message won't be shown*")
    assert result.ret == 1

def test_log_report_captures_according_to_config_option_upon_failure(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    'Test that upon failure:\n    (1) `caplog` succeeded to capture the DEBUG message and assert on it => No `Exception` is raised.\n    (2) The `DEBUG` message does NOT appear in the `Captured log call` report.\n    (3) The stdout, `INFO`, and `WARNING` messages DO appear in the test reports due to `--log-level=INFO`.\n    '
    pytester.makepyfile("\n        import pytest\n        import logging\n\n        def function_that_logs():\n            logging.debug('DEBUG log ' + 'message')\n            logging.info('INFO log ' + 'message')\n            logging.warning('WARNING log ' + 'message')\n            print('Print ' + 'message')\n\n        def test_that_fails(request, caplog):\n            plugin = request.config.pluginmanager.getplugin('logging-plugin')\n            assert plugin.log_level == logging.INFO\n\n            with caplog.at_level(logging.DEBUG):\n                function_that_logs()\n\n            if 'DEBUG log ' + 'message' not in caplog.text:\n                raise Exception('caplog failed to ' + 'capture DEBUG')\n\n            assert False\n    ")
    result = pytester.runpytest('--log-level=INFO')
    result.stdout.no_fnmatch_line('*Exception: caplog failed to capture DEBUG*')
    result.stdout.no_fnmatch_line('*DEBUG log message*')
    result.stdout.fnmatch_lines(['*Print message*', '*INFO log message*', '*WARNING log message*'])
    assert result.ret == 1