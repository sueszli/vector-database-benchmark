from __future__ import absolute_import
import unittest2
from st2common.util.compat import to_ascii
__all__ = ['CompatUtilsTestCase']

class CompatUtilsTestCase(unittest2.TestCase):

    def test_to_ascii(self):
        if False:
            return 10
        expected_values = [('already ascii', 'already ascii'), ('foo', 'foo'), ('٩(̾●̮̮̃̾•̃̾)۶', '()'), ('Ù©', '')]
        for (input_value, expected_value) in expected_values:
            result = to_ascii(input_value)
            self.assertEqual(result, expected_value)