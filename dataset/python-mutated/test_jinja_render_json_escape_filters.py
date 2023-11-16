from __future__ import absolute_import
import unittest2
from st2common.util import jinja as jinja_utils

class JinjaUtilsJsonEscapeTestCase(unittest2.TestCase):

    def test_doublequotes(self):
        if False:
            return 10
        env = jinja_utils.get_jinja_environment()
        template = '{{ test_str | json_escape }}'
        actual = env.from_string(template).render({'test_str': 'foo """ bar'})
        expected = 'foo \\"\\"\\" bar'
        self.assertEqual(actual, expected)

    def test_backslashes(self):
        if False:
            print('Hello World!')
        env = jinja_utils.get_jinja_environment()
        template = '{{ test_str | json_escape }}'
        actual = env.from_string(template).render({'test_str': 'foo \\ bar'})
        expected = 'foo \\\\ bar'
        self.assertEqual(actual, expected)

    def test_backspace(self):
        if False:
            i = 10
            return i + 15
        env = jinja_utils.get_jinja_environment()
        template = '{{ test_str | json_escape }}'
        actual = env.from_string(template).render({'test_str': 'foo \x08 bar'})
        expected = 'foo \\b bar'
        self.assertEqual(actual, expected)

    def test_formfeed(self):
        if False:
            for i in range(10):
                print('nop')
        env = jinja_utils.get_jinja_environment()
        template = '{{ test_str | json_escape }}'
        actual = env.from_string(template).render({'test_str': 'foo \x0c bar'})
        expected = 'foo \\f bar'
        self.assertEqual(actual, expected)

    def test_newline(self):
        if False:
            for i in range(10):
                print('nop')
        env = jinja_utils.get_jinja_environment()
        template = '{{ test_str | json_escape }}'
        actual = env.from_string(template).render({'test_str': 'foo \n bar'})
        expected = 'foo \\n bar'
        self.assertEqual(actual, expected)

    def test_carriagereturn(self):
        if False:
            while True:
                i = 10
        env = jinja_utils.get_jinja_environment()
        template = '{{ test_str | json_escape }}'
        actual = env.from_string(template).render({'test_str': 'foo \r bar'})
        expected = 'foo \\r bar'
        self.assertEqual(actual, expected)

    def test_tab(self):
        if False:
            i = 10
            return i + 15
        env = jinja_utils.get_jinja_environment()
        template = '{{ test_str | json_escape }}'
        actual = env.from_string(template).render({'test_str': 'foo \t bar'})
        expected = 'foo \\t bar'
        self.assertEqual(actual, expected)