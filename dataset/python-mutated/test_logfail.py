"""Tests for the LogFailHandler test helper."""
import logging
import pytest

def test_log_debug():
    if False:
        for i in range(10):
            print('nop')
    logging.debug('foo')

def test_log_warning():
    if False:
        print('Hello World!')
    with pytest.raises(pytest.fail.Exception):
        logging.warning('foo')

def test_log_expected(caplog):
    if False:
        print('Hello World!')
    with caplog.at_level(logging.ERROR):
        logging.error('foo')

def test_log_expected_logger(caplog):
    if False:
        i = 10
        return i + 15
    logger = 'logfail_test_logger'
    with caplog.at_level(logging.ERROR, logger):
        logging.getLogger(logger).error('foo')

def test_log_expected_wrong_level(caplog):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(pytest.fail.Exception):
        with caplog.at_level(logging.ERROR):
            logging.critical('foo')

def test_log_expected_logger_wrong_level(caplog):
    if False:
        while True:
            i = 10
    logger = 'logfail_test_logger'
    with pytest.raises(pytest.fail.Exception):
        with caplog.at_level(logging.ERROR, logger):
            logging.getLogger(logger).critical('foo')

def test_log_expected_wrong_logger(caplog):
    if False:
        print('Hello World!')
    logger = 'logfail_test_logger'
    with pytest.raises(pytest.fail.Exception):
        with caplog.at_level(logging.ERROR, logger):
            logging.error('foo')