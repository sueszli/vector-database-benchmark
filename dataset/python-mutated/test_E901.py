"""moved from data files due to 3.12 changing syntax errors"""
import sys
import unittest
from testing.support import errors_from_src

class E901Test(unittest.TestCase):

    def test_closing_brace(self):
        if False:
            for i in range(10):
                print('nop')
        errors = errors_from_src('}\n')
        if sys.version_info < (3, 12):
            self.assertEqual(errors, ['E901:2:1'])
        else:
            self.assertEqual(errors, [])

    def test_unclosed_brace(self):
        if False:
            return 10
        src = "if msg:\n    errmsg = msg % progress.get(cr_dbname))\n\ndef lasting(self, duration=300):\n    progress = self._progress.setdefault('foo', {}\n"
        errors = errors_from_src(src)
        if sys.version_info < (3, 12):
            expected = ['E122:4:1', 'E251:5:13', 'E251:5:15']
        else:
            expected = ['E122:4:1', 'E251:5:13', 'E251:5:15', 'E901:5:1']
        self.assertEqual(errors, expected)