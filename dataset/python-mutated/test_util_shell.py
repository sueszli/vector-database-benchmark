from __future__ import absolute_import
import unittest2
from st2common.util.shell import quote_unix
from st2common.util.shell import quote_windows
from six.moves import zip

class ShellUtilsTestCase(unittest2.TestCase):

    def test_quote_unix(self):
        if False:
            print('Hello World!')
        arguments = ['foo', 'foo bar', 'foo1 bar1', '"foo"', '"foo" "bar"', "'foo bar'"]
        expected_values = ['\n            foo\n            ', "\n            'foo bar'\n            ", "\n            'foo1 bar1'\n            ", '\n            \'"foo"\'\n            ', '\n            \'"foo" "bar"\'\n            ', '\n            \'\'"\'"\'foo bar\'"\'"\'\'\n            ']
        for (argument, expected_value) in zip(arguments, expected_values):
            actual_value = quote_unix(value=argument)
            expected_value = expected_value.lstrip()
            self.assertEqual(actual_value, expected_value.strip())

    def test_quote_windows(self):
        if False:
            i = 10
            return i + 15
        arguments = ['foo', 'foo bar', 'foo1 bar1', '"foo"', '"foo" "bar"', "'foo bar'"]
        expected_values = ['\n            foo\n            ', '\n            "foo bar"\n            ', '\n            "foo1 bar1"\n            ', '\n            \\"foo\\"\n            ', '\n            "\\"foo\\" \\"bar\\""\n            ', '\n            "\'foo bar\'"\n            ']
        for (argument, expected_value) in zip(arguments, expected_values):
            actual_value = quote_windows(value=argument)
            expected_value = expected_value.lstrip()
            self.assertEqual(actual_value, expected_value.strip())