from __future__ import absolute_import
from __future__ import print_function
import unittest

def patch():
    if False:
        i = 10
        return i + 15
    hasAssertRaisesRegex = getattr(unittest.TestCase, 'assertRaisesRegex', None)
    if not hasAssertRaisesRegex:
        unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp