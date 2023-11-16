import logging
import pytest
from mock import ANY, Mock, call, patch
from nameko.log_helpers import make_timing_logger

@pytest.fixture
def mock_logging():
    if False:
        print('Hello World!')
    with patch('nameko.log_helpers.logging') as patched:
        yield patched

@pytest.fixture
def mock_time():
    if False:
        i = 10
        return i + 15
    with patch('nameko.log_helpers.time') as patched:
        patched.time.side_effect = [0, 0.123456789]
        yield

def test_timing_logger(mock_logging, mock_time):
    if False:
        while True:
            i = 10
    logger = Mock()
    log_time = make_timing_logger(logger)
    with log_time('msg %s', 'foo'):
        pass
    assert logger.log.call_args_list == [call(logging.DEBUG, 'msg %s in %0.3fs', 'foo', 0.123456789)]

def test_timing_logger_level(mock_logging):
    if False:
        return 10
    logger = Mock()
    log_time = make_timing_logger(logger, level=logging.INFO)
    with log_time('msg %s', 'foo'):
        pass
    assert logger.log.call_args_list == [call(logging.INFO, 'msg %s in %0.3fs', 'foo', ANY)]

def test_timing_logger_precision(mock_logging):
    if False:
        print('Hello World!')
    logger = Mock()
    log_time = make_timing_logger(logger, precision=5)
    with log_time('msg %s', 'foo'):
        pass
    assert logger.log.call_args_list == [call(logging.DEBUG, 'msg %s in %0.5fs', 'foo', ANY)]