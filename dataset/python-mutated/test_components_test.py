"""Tests for the test_components module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import test_components as tc
from fire import testutils

class TestComponentsTest(testutils.BaseTestCase):
    """Tests to verify that the test components are importable and okay."""

    def testTestComponents(self):
        if False:
            print('Hello World!')
        self.assertIsNotNone(tc.Empty)
        self.assertIsNotNone(tc.OldStyleEmpty)

    def testNonComparable(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            tc.NonComparable() != 2
        with self.assertRaises(ValueError):
            tc.NonComparable() == 2
if __name__ == '__main__':
    testutils.main()