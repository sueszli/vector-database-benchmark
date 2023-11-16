"""
Tests for the blank_lines checker.

It uses dedicated assertions which work with TestReport.
"""
import unittest
import pycodestyle
from testing.support import errors_from_src

class BlankLinesTestCase(unittest.TestCase):
    """
    Common code for running blank_lines tests.
    """

    def assertNoErrors(self, actual):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that the actual result from the checker has no errors.\n        '
        self.assertEqual([], actual)

class TestBlankLinesDefault(BlankLinesTestCase):
    """
    Tests for default blank with 2 blank lines for top level and 1
    blank line for methods.
    """

    def test_initial_no_blank(self):
        if False:
            return 10
        '\n        It will accept no blank lines at the start of the file.\n        '
        result = errors_from_src('def some_function():\n    pass\n')
        self.assertNoErrors(result)

    def test_initial_lines_one_blank(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        It will accept 1 blank lines before the first line of actual\n        code, even if in other places it asks for 2\n        '
        result = errors_from_src('\ndef some_function():\n    pass\n')
        self.assertNoErrors(result)

    def test_initial_lines_two_blanks(self):
        if False:
            return 10
        '\n        It will accept 2 blank lines before the first line of actual\n        code, as normal.\n        '
        result = errors_from_src('\n\ndef some_function():\n    pass\n')
        self.assertNoErrors(result)

    def test_method_less_blank_lines(self):
        if False:
            print('Hello World!')
        '\n        It will trigger an error when less than 1 blank lin is found\n        before method definitions.\n        '
        result = errors_from_src('# First comment line.\nclass X:\n\n    def a():\n        pass\n    def b():\n        pass\n')
        self.assertEqual(['E301:6:5'], result)

    def test_method_less_blank_lines_comment(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        It will trigger an error when less than 1 blank lin is found\n        before method definition, ignoring comments.\n        '
        result = errors_from_src('# First comment line.\nclass X:\n\n    def a():\n        pass\n    # A comment will not make it better.\n    def b():\n        pass\n')
        self.assertEqual(['E301:7:5'], result)

    def test_top_level_fewer_blank_lines(self):
        if False:
            print('Hello World!')
        '\n        It will trigger an error when less 2 blank lines are found\n        before top level definitions.\n        '
        result = errors_from_src('# First comment line.\n# Second line of comment.\n\ndef some_function():\n    pass\n\nasync def another_function():\n    pass\n\n\ndef this_one_is_good():\n    pass\n\nclass SomeCloseClass(object):\n    pass\n\n\nasync def this_async_is_good():\n    pass\n\n\nclass AFarEnoughClass(object):\n    pass\n')
        self.assertEqual(['E302:7:1', 'E302:14:1'], result)

    def test_top_level_more_blank_lines(self):
        if False:
            print('Hello World!')
        '\n        It will trigger an error when more 2 blank lines are found\n        before top level definitions.\n        '
        result = errors_from_src('# First comment line.\n# Second line of comment.\n\n\n\ndef some_function():\n    pass\n\n\ndef this_one_is_good():\n    pass\n\n\n\nclass SomeFarClass(object):\n    pass\n\n\nclass AFarEnoughClass(object):\n    pass\n')
        self.assertEqual(['E303:6:1', 'E303:15:1'], result)

    def test_method_more_blank_lines(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        It will trigger an error when more than 1 blank line is found\n        before method definition\n        '
        result = errors_from_src('# First comment line.\n\n\nclass SomeCloseClass(object):\n\n\n    def oneMethod(self):\n        pass\n\n\n    def anotherMethod(self):\n        pass\n\n    def methodOK(self):\n        pass\n\n\n\n    def veryFar(self):\n        pass\n')
        self.assertEqual(['E303:7:5', 'E303:11:5', 'E303:19:5'], result)

    def test_initial_lines_more_blank(self):
        if False:
            print('Hello World!')
        '\n        It will trigger an error for more than 2 blank lines before the\n        first line of actual code.\n        '
        result = errors_from_src('\n\n\ndef some_function():\n    pass\n')
        self.assertEqual(['E303:4:1'], result)

    def test_blank_line_between_decorator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        It will trigger an error when the decorator is followed by a\n        blank line.\n        '
        result = errors_from_src('# First line.\n\n\n@some_decorator\n\ndef some_function():\n    pass\n\n\nclass SomeClass(object):\n\n    @method_decorator\n\n    def some_method(self):\n        pass\n')
        self.assertEqual(['E304:6:1', 'E304:14:5'], result)

    def test_blank_line_decorator(self):
        if False:
            return 10
        '\n        It will accept the decorators which are adjacent to the function\n        and method definition.\n        '
        result = errors_from_src('# First line.\n\n\n@another_decorator\n@some_decorator\ndef some_function():\n    pass\n\n\nclass SomeClass(object):\n\n    @method_decorator\n    def some_method(self):\n        pass\n')
        self.assertNoErrors(result)

    def test_top_level_fewer_follow_lines(self):
        if False:
            while True:
                i = 10
        '\n        It will trigger an error when less than 2 blank lines are\n        found between a top level definitions and other top level code.\n        '
        result = errors_from_src("\ndef a():\n    print('Something')\n\na()\n")
        self.assertEqual(['E305:5:1'], result)

    def test_top_level_fewer_follow_lines_comments(self):
        if False:
            while True:
                i = 10
        '\n        It will trigger an error when less than 2 blank lines are\n        found between a top level definitions and other top level code,\n        even if we have comments before\n        '
        result = errors_from_src("\ndef a():\n    print('Something')\n\n    # comment\n\n    # another comment\n\n# With comment still needs 2 spaces before,\n# as comments are ignored.\na()\n")
        self.assertEqual(['E305:11:1'], result)

    def test_top_level_good_follow_lines(self):
        if False:
            while True:
                i = 10
        '\n        It not trigger an error when 2 blank lines are\n        found between a top level definitions and other top level code.\n        '
        result = errors_from_src("\ndef a():\n    print('Something')\n\n    # Some comments in other parts.\n\n    # More comments.\n\n\n# With the right spaces,\n# It will work, even when we have comments.\na()\n")
        self.assertNoErrors(result)

    def test_method_fewer_follow_lines(self):
        if False:
            while True:
                i = 10
        '\n        It will trigger an error when less than 1 blank line is\n        found between a method and previous definitions.\n        '
        result = errors_from_src('\ndef a():\n    x = 1\n    def b():\n        pass\n')
        self.assertEqual(['E306:4:5'], result)

    def test_method_nested_fewer_follow_lines(self):
        if False:
            print('Hello World!')
        '\n        It will trigger an error when less than 1 blank line is\n        found between a method and previous definitions, even when\n        nested.\n        '
        result = errors_from_src('\ndef a():\n    x = 2\n\n    def b():\n        x = 1\n        def c():\n            pass\n')
        self.assertEqual(['E306:7:9'], result)

    def test_method_nested_less_class(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        It will trigger an error when less than 1 blank line is found\n        between a method and previous definitions, even when used to\n        define a class.\n        '
        result = errors_from_src('\ndef a():\n    x = 1\n    class C:\n        pass\n')
        self.assertEqual(['E306:4:5'], result)

    def test_method_nested_ok(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Will not trigger an error when 1 blank line is found\n        found between a method and previous definitions, even when\n        nested.\n        '
        result = errors_from_src('\ndef a():\n    x = 2\n\n    def b():\n        x = 1\n\n        def c():\n            pass\n\n    class C:\n        pass\n')
        self.assertNoErrors(result)

class TestBlankLinesTwisted(BlankLinesTestCase):
    """
    Tests for blank_lines with 3 blank lines for top level and 2 blank
    line for methods as used by the Twisted coding style.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._original_lines_config = pycodestyle.BLANK_LINES_CONFIG.copy()
        pycodestyle.BLANK_LINES_CONFIG['top_level'] = 3
        pycodestyle.BLANK_LINES_CONFIG['method'] = 2

    def tearDown(self):
        if False:
            return 10
        pycodestyle.BLANK_LINES_CONFIG = self._original_lines_config

    def test_initial_lines_one_blanks(self):
        if False:
            while True:
                i = 10
        '\n        It will accept less than 3 blank lines before the first line of\n        actual code.\n        '
        result = errors_from_src('\n\n\ndef some_function():\n    pass\n')
        self.assertNoErrors(result)

    def test_initial_lines_tree_blanks(self):
        if False:
            while True:
                i = 10
        '\n        It will accept 3 blank lines before the first line of actual\n        code, as normal.\n        '
        result = errors_from_src('\n\n\ndef some_function():\n    pass\n')
        self.assertNoErrors(result)

    def test_top_level_fewer_blank_lines(self):
        if False:
            return 10
        '\n        It will trigger an error when less 3 blank lines are found\n        before top level definitions.\n        '
        result = errors_from_src('# First comment line.\n# Second line of comment.\n\n\ndef some_function():\n    pass\n\n\nasync def another_function():\n    pass\n\n\n\ndef this_one_is_good():\n    pass\n\nclass SomeCloseClass(object):\n    pass\n\n\n\nasync def this_async_is_good():\n    pass\n\n\n\nclass AFarEnoughClass(object):\n    pass\n')
        self.assertEqual(['E302:9:1', 'E302:17:1'], result)

    def test_top_level_more_blank_lines(self):
        if False:
            i = 10
            return i + 15
        '\n        It will trigger an error when more 2 blank lines are found\n        before top level definitions.\n        '
        result = errors_from_src('# First comment line.\n# Second line of comment.\n\n\n\n\ndef some_function():\n    pass\n\n\n\ndef this_one_is_good():\n    pass\n\n\n\n\nclass SomeVeryFarClass(object):\n    pass\n\n\n\nclass AFarEnoughClass(object):\n    pass\n')
        self.assertEqual(['E303:7:1', 'E303:18:1'], result)

    def test_the_right_blanks(self):
        if False:
            i = 10
            return i + 15
        '\n        It will accept 3 blank for top level and 2 for nested.\n        '
        result = errors_from_src("\n\n\ndef some_function():\n    pass\n\n\n\n# With comments.\nsome_other = code_here\n\n\n\nclass SomeClass:\n    '''\n    Docstring here.\n    '''\n\n    def some_method():\n        pass\n\n\n    def another_method():\n        pass\n\n\n    # More methods.\n    def another_method_with_comment():\n        pass\n\n\n    @decorator\n    def another_method_with_comment():\n        pass\n")
        self.assertNoErrors(result)