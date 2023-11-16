from __future__ import absolute_import
import unittest2
from st2common.util import jinja as jinja_utils

class JinjaUtilsVersionsFilterTestCase(unittest2.TestCase):

    def test_version_compare(self):
        if False:
            while True:
                i = 10
        env = jinja_utils.get_jinja_environment()
        template = '{{version | version_compare("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.9.0'})
        expected = '-1'
        self.assertEqual(actual, expected)
        template = '{{version | version_compare("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = '1'
        self.assertEqual(actual, expected)
        template = '{{version | version_compare("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.0'})
        expected = '0'
        self.assertEqual(actual, expected)

    def test_version_more_than(self):
        if False:
            i = 10
            return i + 15
        env = jinja_utils.get_jinja_environment()
        template = '{{version | version_more_than("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.9.0'})
        expected = 'False'
        self.assertEqual(actual, expected)
        template = '{{version | version_more_than("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = 'True'
        self.assertEqual(actual, expected)
        template = '{{version | version_more_than("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.0'})
        expected = 'False'
        self.assertEqual(actual, expected)

    def test_version_less_than(self):
        if False:
            for i in range(10):
                print('nop')
        env = jinja_utils.get_jinja_environment()
        template = '{{version | version_less_than("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.9.0'})
        expected = 'True'
        self.assertEqual(actual, expected)
        template = '{{version | version_less_than("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = 'False'
        self.assertEqual(actual, expected)
        template = '{{version | version_less_than("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.0'})
        expected = 'False'
        self.assertEqual(actual, expected)

    def test_version_equal(self):
        if False:
            print('Hello World!')
        env = jinja_utils.get_jinja_environment()
        template = '{{version | version_equal("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.9.0'})
        expected = 'False'
        self.assertEqual(actual, expected)
        template = '{{version | version_equal("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = 'False'
        self.assertEqual(actual, expected)
        template = '{{version | version_equal("0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.0'})
        expected = 'True'
        self.assertEqual(actual, expected)

    def test_version_match(self):
        if False:
            print('Hello World!')
        env = jinja_utils.get_jinja_environment()
        template = '{{version | version_match(">0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = 'True'
        self.assertEqual(actual, expected)
        actual = env.from_string(template).render({'version': '0.1.1'})
        expected = 'False'
        self.assertEqual(actual, expected)
        template = '{{version | version_match("<0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.1.0'})
        expected = 'True'
        self.assertEqual(actual, expected)
        actual = env.from_string(template).render({'version': '1.1.0'})
        expected = 'False'
        self.assertEqual(actual, expected)
        template = '{{version | version_match("==0.10.0")}}'
        actual = env.from_string(template).render({'version': '0.10.0'})
        expected = 'True'
        self.assertEqual(actual, expected)
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = 'False'
        self.assertEqual(actual, expected)

    def test_version_bump_major(self):
        if False:
            for i in range(10):
                print('nop')
        env = jinja_utils.get_jinja_environment()
        template = '{{version | version_bump_major}}'
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = '1.0.0'
        self.assertEqual(actual, expected)

    def test_version_bump_minor(self):
        if False:
            i = 10
            return i + 15
        env = jinja_utils.get_jinja_environment()
        template = '{{version | version_bump_minor}}'
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = '0.11.0'
        self.assertEqual(actual, expected)

    def test_version_bump_patch(self):
        if False:
            return 10
        env = jinja_utils.get_jinja_environment()
        template = '{{version | version_bump_patch}}'
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = '0.10.2'
        self.assertEqual(actual, expected)

    def test_version_strip_patch(self):
        if False:
            i = 10
            return i + 15
        env = jinja_utils.get_jinja_environment()
        template = '{{version | version_strip_patch}}'
        actual = env.from_string(template).render({'version': '0.10.1'})
        expected = '0.10'
        self.assertEqual(actual, expected)