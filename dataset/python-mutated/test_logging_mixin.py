from __future__ import annotations
import logging
import sys
import warnings
from unittest import mock
import pytest
from airflow.utils.log.logging_mixin import LoggingMixin, SetContextPropagate, StreamLogWriter, set_context

@pytest.fixture
def logger():
    if False:
        for i in range(10):
            print('nop')
    parent = logging.getLogger(__name__)
    parent.propagate = False
    yield parent
    parent.propagate = True

@pytest.fixture
def child_logger(logger):
    if False:
        return 10
    yield logger.getChild('child')

@pytest.fixture
def parent_child_handlers(child_logger):
    if False:
        while True:
            i = 10
    parent_handler = logging.NullHandler()
    parent_handler.handle = mock.MagicMock(name='parent_handler.handle')
    child_handler = logging.NullHandler()
    child_handler.handle = mock.MagicMock(name='handler.handle')
    logger = child_logger.parent
    logger.addHandler(parent_handler)
    child_logger.addHandler(child_handler)
    child_logger.propagate = True
    yield (parent_handler, child_handler)
    logger.removeHandler(parent_handler)
    child_logger.removeHandler(child_handler)

class TestLoggingMixin:

    def setup_method(self):
        if False:
            print('Hello World!')
        warnings.filterwarnings(action='always')

    def test_set_context(self, child_logger, parent_child_handlers):
        if False:
            return 10
        (handler1, handler2) = parent_child_handlers
        handler1.set_context = mock.MagicMock()
        handler2.set_context = mock.MagicMock()
        parent = logging.getLogger(__name__)
        parent.propagate = False
        parent.addHandler(handler1)
        log = parent.getChild('child')
        log.addHandler(handler2)
        log.propagate = True
        value = 'test'
        set_context(log, value)
        handler1.set_context.assert_called_once_with(value)
        handler2.set_context.assert_called_once_with(value)

    def test_default_logger_name(self):
        if False:
            while True:
                i = 10
        '\n        Ensure that by default, object logger name is equals to its module and class path.\n        '

        class DummyClass(LoggingMixin):
            pass
        assert DummyClass().log.name == 'tests.utils.test_logging_mixin.DummyClass'

    def test_logger_name_is_root_when_logger_name_is_empty_string(self):
        if False:
            print('Hello World!')
        "\n        Ensure that when `_logger_name` is set as an empty string, the resulting logger name is an empty\n        string too, which result in a logger with 'root' as name.\n        Note: Passing an empty string to `logging.getLogger` will create a logger with name 'root'.\n        "

        class EmptyStringLogger(LoggingMixin):
            _logger_name: str | None = ''
        assert EmptyStringLogger().log.name == 'root'

    def test_log_config_logger_name_correctly_prefix_logger_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure that when a class has `_log_config_logger_name`, it is used as prefix in the final logger\n        name.\n        '

        class ClassWithParentLogConfig(LoggingMixin):
            _log_config_logger_name: str = 'airflow.tasks'
        assert ClassWithParentLogConfig().log.name == 'airflow.tasks.tests.utils.test_logging_mixin.ClassWithParentLogConfig'

    def teardown_method(self):
        if False:
            return 10
        warnings.resetwarnings()

class TestStreamLogWriter:

    def test_write(self):
        if False:
            for i in range(10):
                print('nop')
        logger = mock.MagicMock()
        logger.log = mock.MagicMock()
        log = StreamLogWriter(logger, 1)
        msg = 'test_message'
        log.write(msg)
        assert log._buffer == msg
        log.write(' \n')
        logger.log.assert_called_once_with(1, msg)
        assert log._buffer == ''

    def test_flush(self):
        if False:
            i = 10
            return i + 15
        logger = mock.MagicMock()
        logger.log = mock.MagicMock()
        log = StreamLogWriter(logger, 1)
        msg = 'test_message'
        log.write(msg)
        assert log._buffer == msg
        log.flush()
        logger.log.assert_called_once_with(1, msg)
        assert log._buffer == ''

    def test_isatty(self):
        if False:
            for i in range(10):
                print('nop')
        logger = mock.MagicMock()
        logger.log = mock.MagicMock()
        log = StreamLogWriter(logger, 1)
        assert not log.isatty()

    def test_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        logger = mock.MagicMock()
        logger.log = mock.MagicMock()
        log = StreamLogWriter(logger, 1)
        assert log.encoding is None

    def test_iobase_compatibility(self):
        if False:
            for i in range(10):
                print('nop')
        log = StreamLogWriter(None, 1)
        assert not log.closed
        log.close()

@pytest.mark.parametrize(['maintain_propagate'], [[SetContextPropagate.MAINTAIN_PROPAGATE], [None]])
def test_set_context_propagation(parent_child_handlers, child_logger, maintain_propagate):
    if False:
        print('Hello World!')
    (parent_handler, handler) = parent_child_handlers
    handler.set_context = mock.MagicMock(return_value=maintain_propagate)
    line = sys._getframe().f_lineno + 1
    record = child_logger.makeRecord(child_logger.name, logging.INFO, __file__, line, 'test message', [], None)
    child_logger.handle(record)
    handler.handle.assert_called_once_with(record)
    parent_handler.handle.assert_called_once_with(record)
    parent_handler.handle.reset_mock()
    handler.handle.reset_mock()
    set_context(child_logger, {})
    child_logger.handle(record)
    handler.handle.assert_called_once_with(record)
    if maintain_propagate is SetContextPropagate.MAINTAIN_PROPAGATE:
        parent_handler.handle.assert_called_once_with(record)
    else:
        parent_handler.handle.assert_not_called()