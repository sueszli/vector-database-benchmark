from __future__ import absolute_import
import unittest2
from st2common.util import jinja as jinja_utils

class JinjaUtilsPathFilterTestCase(unittest2.TestCase):

    def test_basename(self):
        if False:
            while True:
                i = 10
        env = jinja_utils.get_jinja_environment()
        template = '{{k1 | basename}}'
        actual = env.from_string(template).render({'k1': '/some/path/to/file.txt'})
        self.assertEqual(actual, 'file.txt')
        actual = env.from_string(template).render({'k1': '/some/path/to/dir'})
        self.assertEqual(actual, 'dir')
        actual = env.from_string(template).render({'k1': '/some/path/to/dir/'})
        self.assertEqual(actual, '')

    def test_dirname(self):
        if False:
            while True:
                i = 10
        env = jinja_utils.get_jinja_environment()
        template = '{{k1 | dirname}}'
        actual = env.from_string(template).render({'k1': '/some/path/to/file.txt'})
        self.assertEqual(actual, '/some/path/to')
        actual = env.from_string(template).render({'k1': '/some/path/to/dir'})
        self.assertEqual(actual, '/some/path/to')
        actual = env.from_string(template).render({'k1': '/some/path/to/dir/'})
        self.assertEqual(actual, '/some/path/to/dir')