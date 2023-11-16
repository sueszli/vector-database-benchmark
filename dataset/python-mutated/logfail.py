"""Logging handling for the tests."""
import logging
import pytest

class LogFailHandler(logging.Handler):
    """A logging handler which makes tests fail on unexpected messages."""

    def __init__(self, level=logging.NOTSET, min_level=logging.WARNING):
        if False:
            return 10
        self._min_level = min_level
        super().__init__(level)

    def emit(self, record):
        if False:
            i = 10
            return i + 15
        logger = logging.getLogger(record.name)
        root_logger = logging.getLogger()
        if logger.name == 'messagemock':
            return
        if record.levelno in (logger.level, root_logger.level):
            return
        if record.levelno < self._min_level:
            return
        pytest.fail('Got logging message on logger {} with level {}: {}!'.format(record.name, record.levelname, record.getMessage()))

@pytest.fixture(scope='session', autouse=True)
def fail_on_logging():
    if False:
        while True:
            i = 10
    handler = LogFailHandler()
    logging.getLogger().addHandler(handler)
    yield
    logging.getLogger().removeHandler(handler)
    handler.close()