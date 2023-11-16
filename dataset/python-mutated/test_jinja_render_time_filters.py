from __future__ import absolute_import
import unittest2
from st2common.util import jinja as jinja_utils

class JinjaUtilsTimeFilterTestCase(unittest2.TestCase):

    def test_to_human_time_filter(self):
        if False:
            print('Hello World!')
        env = jinja_utils.get_jinja_environment()
        template = '{{k1 | to_human_time_from_seconds}}'
        actual = env.from_string(template).render({'k1': 12345})
        self.assertEqual(actual, '3h25m45s')
        actual = env.from_string(template).render({'k1': 0})
        self.assertEqual(actual, '0s')
        self.assertRaises(AssertionError, env.from_string(template).render, {'k1': 'stuff'})