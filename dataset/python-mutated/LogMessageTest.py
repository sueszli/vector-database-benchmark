import unittest
from datetime import datetime
from coalib.misc import Constants
from coalib.output.printers.LOG_LEVEL import LOG_LEVEL
from coalib.processes.communication.LogMessage import LogMessage

class LogMessageTest(unittest.TestCase):
    timestamp = datetime.today()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.uut = LogMessage(LOG_LEVEL.DEBUG, 'test', 'message', timestamp=self.timestamp)

    def test_construction(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.uut.log_level, LOG_LEVEL.DEBUG)
        self.assertEqual(self.uut.message, 'test message')
        self.assertEqual(self.uut.timestamp, self.timestamp)
        self.uut = LogMessage(LOG_LEVEL.WARNING, '   a msg  ', 5, '  ', timestamp=self.timestamp)
        self.assertEqual(self.uut.log_level, LOG_LEVEL.WARNING)
        self.assertEqual(self.uut.message, '   a msg   5')
        self.assertEqual(self.uut.timestamp, self.timestamp)
        self.assertRaises(ValueError, LogMessage, LOG_LEVEL.DEBUG, '')
        self.assertRaises(ValueError, LogMessage, 5, 'test')

    def test_to_str(self):
        if False:
            return 10
        self.uut.message = Constants.COMPLEX_TEST_STRING
        self.uut.log_level = LOG_LEVEL.ERROR
        self.assertEqual(str(self.uut), f'[ERROR] {Constants.COMPLEX_TEST_STRING}')
        self.uut.log_level = LOG_LEVEL.WARNING
        self.assertEqual(str(self.uut), f'[WARNING] {Constants.COMPLEX_TEST_STRING}')
        self.uut.log_level = LOG_LEVEL.DEBUG
        self.assertEqual(str(self.uut), f'[DEBUG] {Constants.COMPLEX_TEST_STRING}')
        self.uut.log_level = 5
        self.assertEqual(str(self.uut), f'[ERROR] {Constants.COMPLEX_TEST_STRING}')

    def test_equals(self):
        if False:
            print('Hello World!')
        self.assertEqual(LogMessage(LOG_LEVEL.DEBUG, 'test message'), LogMessage(LOG_LEVEL.DEBUG, 'test message'))
        self.assertNotEqual(LogMessage(LOG_LEVEL.DEBUG, 'test message'), LogMessage(LOG_LEVEL.WARNING, 'test message'))
        self.assertNotEqual(LogMessage(LOG_LEVEL.DEBUG, 'test message'), LogMessage(LOG_LEVEL.DEBUG, 'test'))
        self.assertNotEqual(LogMessage(LOG_LEVEL.DEBUG, 'test message'), 5)

    def test_string_dict(self):
        if False:
            for i in range(10):
                print('nop')
        self.uut.log_level = LOG_LEVEL.DEBUG
        self.uut.message = 'test'
        self.assertEqual(self.uut.to_string_dict(), {'log_level': 'DEBUG', 'message': 'test', 'timestamp': self.timestamp.isoformat()})
        self.uut.timestamp = None
        self.uut.log_level = -9999
        self.assertEqual(self.uut.to_string_dict(), {'log_level': '', 'message': 'test', 'timestamp': ''})