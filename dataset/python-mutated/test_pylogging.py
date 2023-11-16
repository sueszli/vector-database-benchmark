import unittest
from robot.utils.asserts import assert_equal
from robot.output.pyloggingconf import RobotHandler
import logging

class MessageMock:

    def __init__(self, timestamp, level, message):
        if False:
            for i in range(10):
                print('nop')
        self.timestamp = timestamp
        self.level = level
        self.message = message

class MockLibraryLogger:

    def __init__(self):
        if False:
            return 10
        self.last_message = (None, None)

    def error(self, message):
        if False:
            i = 10
            return i + 15
        self.last_message = (message, logging.ERROR)

    def warn(self, message):
        if False:
            i = 10
            return i + 15
        self.last_message = (message, logging.WARNING)

    def info(self, message):
        if False:
            i = 10
            return i + 15
        self.last_message = (message, logging.INFO)

    def debug(self, message):
        if False:
            print('Hello World!')
        self.last_message = (message, logging.DEBUG)

    def trace(self, message):
        if False:
            while True:
                i = 10
        self.last_message = (message, logging.NOTSET)

class TestPyLogging(unittest.TestCase):
    test_message = 'This is a test message'
    test_format = '%(name)s %(levelname)s %(message)s'
    str_rep = {logging.ERROR: 'ERROR', logging.WARNING: 'WARNING', logging.INFO: 'INFO', logging.DEBUG: 'DEBUG', logging.NOTSET: 'TRACE'}

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.library_logger = MockLibraryLogger()
        self.test_handler = RobotHandler(library_logger=self.library_logger)
        root = logging.getLogger()
        root.setLevel(logging.NOTSET)
        for handler in root.handlers:
            root.removeHandler(handler)
        root.addHandler(self.test_handler)

    def tearDown(self):
        if False:
            return 10
        root = logging.getLogger()
        root.removeHandler(self.test_handler)

    def test_python_logging_debug(self):
        if False:
            print('Hello World!')
        logging.debug(self.test_message)
        self.assert_message(self.test_message, logging.DEBUG)

    def test_python_logging_info(self):
        if False:
            i = 10
            return i + 15
        logging.info(self.test_message)
        self.assert_message(self.test_message, logging.INFO)

    def test_python_logging_warn(self):
        if False:
            for i in range(10):
                print('nop')
        logging.warning(self.test_message)
        self.assert_message(self.test_message, logging.WARNING)

    def test_python_logging_error(self):
        if False:
            while True:
                i = 10
        logging.error(self.test_message)
        self.assert_message(self.test_message, logging.ERROR)

    def test_python_logging_formatted_debug(self):
        if False:
            i = 10
            return i + 15
        old_formatter = self.test_handler.formatter
        formatter = logging.Formatter(fmt=self.test_format)
        self.test_handler.setFormatter(formatter)
        logging.debug(self.test_message)
        self.assert_formatted_message(logging.DEBUG)
        self.test_handler.setFormatter(old_formatter)

    def test_python_logging_formatted_info(self):
        if False:
            return 10
        old_formatter = self.test_handler.formatter
        formatter = logging.Formatter(fmt=self.test_format)
        self.test_handler.setFormatter(formatter)
        logging.info(self.test_message)
        self.assert_formatted_message(logging.INFO)
        self.test_handler.setFormatter(old_formatter)

    def test_python_logging_formatted_warn(self):
        if False:
            for i in range(10):
                print('nop')
        old_formatter = self.test_handler.formatter
        formatter = logging.Formatter(fmt=self.test_format)
        self.test_handler.setFormatter(formatter)
        logging.warning(self.test_message)
        self.assert_formatted_message(logging.WARNING)
        self.test_handler.setFormatter(old_formatter)

    def test_python_logging_formatted_error(self):
        if False:
            print('Hello World!')
        old_formatter = self.test_handler.formatter
        formatter = logging.Formatter(fmt=self.test_format)
        self.test_handler.setFormatter(formatter)
        logging.error(self.test_message)
        self.assert_formatted_message(logging.ERROR)
        self.test_handler.setFormatter(old_formatter)

    def assert_message(self, message, level):
        if False:
            print('Hello World!')
        (message_last, level_last) = self.library_logger.last_message
        assert_equal(message_last, message)
        assert_equal(level_last, level)

    def assert_formatted_message(self, logging_level):
        if False:
            for i in range(10):
                print('nop')
        formatted_message = f'root {self.str_rep[logging_level]} {self.test_message}'
        self.assert_message(formatted_message, logging_level)