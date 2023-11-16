"""Test the test utilities for Fire's tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from fire import testutils
import six

class TestTestUtils(testutils.BaseTestCase):
    """Let's get meta."""

    def testNoCheckOnException(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            with self.assertOutputMatches(stdout='blah'):
                raise ValueError()

    def testCheckStdoutOrStderrNone(self):
        if False:
            print('Hello World!')
        with six.assertRaisesRegex(self, AssertionError, 'stdout:'):
            with self.assertOutputMatches(stdout=None):
                print('blah')
        with six.assertRaisesRegex(self, AssertionError, 'stderr:'):
            with self.assertOutputMatches(stderr=None):
                print('blah', file=sys.stderr)
        with six.assertRaisesRegex(self, AssertionError, 'stderr:'):
            with self.assertOutputMatches(stdout='apple', stderr=None):
                print('apple')
                print('blah', file=sys.stderr)

    def testCorrectOrderingOfAssertRaises(self):
        if False:
            return 10
        with self.assertOutputMatches(stdout='Yep.*first.*second'):
            with self.assertRaises(ValueError):
                print('Yep, this is the first line.\nThis is the second.')
                raise ValueError()
if __name__ == '__main__':
    testutils.main()