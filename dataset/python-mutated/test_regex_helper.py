import re
import unittest
from django.test import SimpleTestCase
from django.utils import regex_helper

class NormalizeTests(unittest.TestCase):

    def test_empty(self):
        if False:
            return 10
        pattern = ''
        expected = [('', [])]
        result = regex_helper.normalize(pattern)
        self.assertEqual(result, expected)

    def test_escape(self):
        if False:
            return 10
        pattern = '\\\\\\^\\$\\.\\|\\?\\*\\+\\(\\)\\['
        expected = [('\\^$.|?*+()[', [])]
        result = regex_helper.normalize(pattern)
        self.assertEqual(result, expected)

    def test_group_positional(self):
        if False:
            while True:
                i = 10
        pattern = '(.*)-(.+)'
        expected = [('%(_0)s-%(_1)s', ['_0', '_1'])]
        result = regex_helper.normalize(pattern)
        self.assertEqual(result, expected)

    def test_group_noncapturing(self):
        if False:
            for i in range(10):
                print('nop')
        pattern = '(?:non-capturing)'
        expected = [('non-capturing', [])]
        result = regex_helper.normalize(pattern)
        self.assertEqual(result, expected)

    def test_group_named(self):
        if False:
            print('Hello World!')
        pattern = '(?P<first_group_name>.*)-(?P<second_group_name>.*)'
        expected = [('%(first_group_name)s-%(second_group_name)s', ['first_group_name', 'second_group_name'])]
        result = regex_helper.normalize(pattern)
        self.assertEqual(result, expected)

    def test_group_backreference(self):
        if False:
            for i in range(10):
                print('nop')
        pattern = '(?P<first_group_name>.*)-(?P=first_group_name)'
        expected = [('%(first_group_name)s-%(first_group_name)s', ['first_group_name'])]
        result = regex_helper.normalize(pattern)
        self.assertEqual(result, expected)

class LazyReCompileTests(SimpleTestCase):

    def test_flags_with_pre_compiled_regex(self):
        if False:
            i = 10
            return i + 15
        test_pattern = re.compile('test')
        lazy_test_pattern = regex_helper._lazy_re_compile(test_pattern, re.I)
        msg = 'flags must be empty if regex is passed pre-compiled'
        with self.assertRaisesMessage(AssertionError, msg):
            lazy_test_pattern.match('TEST')