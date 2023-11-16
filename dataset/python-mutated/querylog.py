import contextlib
import logging
from twisted.python import log

class _QueryToTwistedHandler(logging.Handler):

    def __init__(self, log_query_result=False, record_mode=False):
        if False:
            print('Hello World!')
        super().__init__()
        self._log_query_result = log_query_result
        self.recordMode = record_mode
        self.records = []

    def emit(self, record):
        if False:
            return 10
        if self.recordMode:
            self.records.append(record.getMessage())
            return
        if record.levelno == logging.DEBUG:
            if self._log_query_result:
                log.msg(f'{record.name}:{record.threadName}:result: {record.getMessage()}')
        else:
            log.msg(f'{record.name}:{record.threadName}:query:  {record.getMessage()}')

def start_log_queries(log_query_result=False, record_mode=False):
    if False:
        return 10
    handler = _QueryToTwistedHandler(log_query_result=log_query_result, record_mode=record_mode)
    logger = logging.getLogger('sqlalchemy.engine')
    handler.prev_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    handler.prev_propagate = logger.propagate
    logger.propagate = False
    return handler

def stop_log_queries(handler):
    if False:
        print('Hello World!')
    assert isinstance(handler, _QueryToTwistedHandler)
    logger = logging.getLogger('sqlalchemy.engine')
    logger.removeHandler(handler)
    logger.propagate = handler.prev_propagate
    logger.setLevel(handler.prev_level)

@contextlib.contextmanager
def log_queries():
    if False:
        print('Hello World!')
    handler = start_log_queries()
    try:
        yield
    finally:
        stop_log_queries(handler)

class SqliteMaxVariableMixin:

    @contextlib.contextmanager
    def assertNoMaxVariables(self):
        if False:
            print('Hello World!')
        handler = start_log_queries(record_mode=True)
        try:
            yield
        finally:
            stop_log_queries(handler)
            for line in handler.records:
                self.assertFalse(line.count('?') > 999, 'too much variables in ' + line)