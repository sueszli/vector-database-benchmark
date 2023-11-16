import sys
from synapse.logging.formatter import LogFormatter
from tests import unittest

class TestException(Exception):
    pass

class LogFormatterTestCase(unittest.TestCase):

    def test_formatter(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        formatter = LogFormatter()
        try:
            raise TestException('testytest')
        except TestException:
            ei = sys.exc_info()
        output = formatter.formatException(ei)
        self.assertIn('testytest', output)
        self.assertIn('Capture point', output)