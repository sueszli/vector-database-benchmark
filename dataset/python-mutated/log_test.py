import unittest
from .. import log

class LogTest(unittest.TestCase):

    def test_truncate(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(log.truncate('a', 10), 'a')
        self.assertEqual(log.truncate('a' * 10, 10), 'a' * 10)
        self.assertEqual(log.truncate('123456789', 4), '1234..[truncated 5 characters]')