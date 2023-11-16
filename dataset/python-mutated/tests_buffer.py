import sys
from unittest import TestCase

class WriteToStdoutStderrTestCase(TestCase):

    def test_pass(self):
        if False:
            i = 10
            return i + 15
        sys.stderr.write('Write to stderr.')
        sys.stdout.write('Write to stdout.')
        self.assertTrue(True)

    def test_fail(self):
        if False:
            return 10
        sys.stderr.write('Write to stderr.')
        sys.stdout.write('Write to stdout.')
        self.assertTrue(False)