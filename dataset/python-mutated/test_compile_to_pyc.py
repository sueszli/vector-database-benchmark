"""Tests for compilation to bytecode."""
from pytype.tests import test_base

class CompileToPycTest(test_base.BaseTest):
    """Tests for compilation to bytecode."""

    def test_compilation_of_unicode_source(self):
        if False:
            while True:
                i = 10
        self.Check("print('←↑→↓')")

    def test_compilation_of_unicode_source_with_encoding(self):
        if False:
            print('Hello World!')
        self.Check("# encoding: utf-8\nprint('←↑→↓')")
        self.Check("#! my/python\n# encoding: utf-8\nprint('←↑→↓')")

    def test_error_line_numbers_with_encoding1(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      # coding: utf-8\n      def foo():\n        return "1".hello  # attribute-error\n    ')

    def test_error_line_numbers_with_encoding2(self):
        if False:
            return 10
        self.CheckWithErrors('\n      #! /bin/python\n      # coding: utf-8\n      def foo():\n        return "1".hello  # attribute-error\n    ')
if __name__ == '__main__':
    test_base.main()