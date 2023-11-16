"""Tests for formatting.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
from fire import testutils
LINE_LENGTH = 80

class FormattingTest(testutils.BaseTestCase):

    def test_bold(self):
        if False:
            for i in range(10):
                print('nop')
        text = formatting.Bold('hello')
        self.assertIn(text, ['hello', '\x1b[1mhello\x1b[0m'])

    def test_underline(self):
        if False:
            while True:
                i = 10
        text = formatting.Underline('hello')
        self.assertIn(text, ['hello', '\x1b[4mhello\x1b[0m'])

    def test_indent(self):
        if False:
            print('Hello World!')
        text = formatting.Indent('hello', spaces=2)
        self.assertEqual('  hello', text)

    def test_indent_multiple_lines(self):
        if False:
            for i in range(10):
                print('nop')
        text = formatting.Indent('hello\nworld', spaces=2)
        self.assertEqual('  hello\n  world', text)

    def test_wrap_one_item(self):
        if False:
            print('Hello World!')
        lines = formatting.WrappedJoin(['rice'])
        self.assertEqual(['rice'], lines)

    def test_wrap_multiple_items(self):
        if False:
            i = 10
            return i + 15
        lines = formatting.WrappedJoin(['rice', 'beans', 'chicken', 'cheese'], width=15)
        self.assertEqual(['rice | beans |', 'chicken |', 'cheese'], lines)

    def test_ellipsis_truncate(self):
        if False:
            while True:
                i = 10
        text = 'This is a string'
        truncated_text = formatting.EllipsisTruncate(text=text, available_space=10, line_length=LINE_LENGTH)
        self.assertEqual('This is...', truncated_text)

    def test_ellipsis_truncate_not_enough_space(self):
        if False:
            print('Hello World!')
        text = 'This is a string'
        truncated_text = formatting.EllipsisTruncate(text=text, available_space=2, line_length=LINE_LENGTH)
        self.assertEqual('This is a string', truncated_text)

    def test_ellipsis_middle_truncate(self):
        if False:
            i = 10
            return i + 15
        text = '1000000000L'
        truncated_text = formatting.EllipsisMiddleTruncate(text=text, available_space=7, line_length=LINE_LENGTH)
        self.assertEqual('10...0L', truncated_text)

    def test_ellipsis_middle_truncate_not_enough_space(self):
        if False:
            i = 10
            return i + 15
        text = '1000000000L'
        truncated_text = formatting.EllipsisMiddleTruncate(text=text, available_space=2, line_length=LINE_LENGTH)
        self.assertEqual('1000000000L', truncated_text)
if __name__ == '__main__':
    testutils.main()