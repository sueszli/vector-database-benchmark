import unittest
import collections
import logging
_LoggingWatcher = collections.namedtuple('_LoggingWatcher', ['records', 'output'])

class _BaseTestCaseContext(object):

    def __init__(self, test_case):
        if False:
            for i in range(10):
                print('nop')
        self.test_case = test_case

    def _raiseFailure(self, standardMsg):
        if False:
            for i in range(10):
                print('nop')
        msg = self.test_case._formatMessage(self.msg, standardMsg)
        raise self.test_case.failureException(msg)

class _CapturingHandler(logging.Handler):
    """
    A logging handler capturing all (raw and formatted) logging output.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        logging.Handler.__init__(self)
        self.watcher = _LoggingWatcher([], [])

    def flush(self):
        if False:
            print('Hello World!')
        pass

    def emit(self, record):
        if False:
            while True:
                i = 10
        self.watcher.records.append(record)
        msg = self.format(record)
        self.watcher.output.append(msg)

class _AssertLogsContext(_BaseTestCaseContext):
    """A context manager used to implement TestCase.assertLogs()."""
    LOGGING_FORMAT = '%(levelname)s:%(name)s:%(message)s'

    def __init__(self, test_case, logger_name, level, assert_logs=True):
        if False:
            return 10
        _BaseTestCaseContext.__init__(self, test_case)
        self.logger_name = logger_name
        self.assert_logs = assert_logs
        if level:
            self.level = logging.getLevelName(level)
        else:
            self.level = logging.DEBUG
        self.msg = None

    def __enter__(self):
        if False:
            while True:
                i = 10
        if isinstance(self.logger_name, logging.Logger):
            logger = self.logger = self.logger_name
        else:
            logger = self.logger = logging.getLogger(self.logger_name)
        formatter = logging.Formatter(self.LOGGING_FORMAT)
        handler = _CapturingHandler()
        handler.setFormatter(formatter)
        self.watcher = handler.watcher
        self.old_handlers = logger.handlers[:]
        self.old_level = logger.level
        self.old_propagate = logger.propagate
        logger.handlers = [handler]
        logger.setLevel(self.level)
        logger.propagate = False
        return handler.watcher

    def __exit__(self, exc_type, exc_value, tb):
        if False:
            i = 10
            return i + 15
        self.logger.handlers = self.old_handlers
        self.logger.propagate = self.old_propagate
        self.logger.setLevel(self.old_level)
        if exc_type is not None:
            return False
        if len(self.watcher.records) == 0 and self.assert_logs:
            self._raiseFailure('no logs of level {} or higher triggered on {}'.format(logging.getLevelName(self.level), self.logger.name))
        elif len(self.watcher.records) > 0 and (not self.assert_logs):
            self._raiseFailure('logs of level {} or higher triggered on {}'.format(logging.getLevelName(self.level), self.logger_name))

class LogTestCase(unittest.TestCase):

    def assertLogs(self, logger=None, level=None):
        if False:
            return 10
        "Fail unless a log message of level *level* or higher is emitted\n        on *logger_name* or its children.  If omitted, *level* defaults to\n        INFO and *logger* defaults to the root logger.\n\n        This method must be used as a context manager, and will yield\n        a recording object with two attributes: `output` and `records`.\n        At the end of the context manager, the `output` attribute will\n        be a list of the matching formatted log messages and the\n        `records` attribute will be a list of the corresponding LogRecord\n        objects.\n\n        Example::\n\n            with self.assertLogs('foo', level='INFO') as cm:\n                logging.getLogger('foo').info('first message')\n                logging.getLogger('foo.bar').error('second message')\n            self.assertEqual(cm.output, ['INFO:foo:first message',\n                                         'ERROR:foo.bar:second message'])\n        "
        return _AssertLogsContext(self, logger, level)

    def assertNoLogs(self, logger=None, level=None):
        if False:
            for i in range(10):
                print('nop')
        return _AssertLogsContext(self, logger, level, assert_logs=False)