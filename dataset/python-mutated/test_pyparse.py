"""Test pyparse, coverage 96%."""
from idlelib import pyparse
import unittest
from collections import namedtuple

class ParseMapTest(unittest.TestCase):

    def test_parsemap(self):
        if False:
            print('Hello World!')
        keepwhite = {ord(c): ord(c) for c in ' \t\n\r'}
        mapping = pyparse.ParseMap(keepwhite)
        self.assertEqual(mapping[ord('\t')], ord('\t'))
        self.assertEqual(mapping[ord('a')], ord('x'))
        self.assertEqual(mapping[1000], ord('x'))

    def test_trans(self):
        if False:
            return 10
        parser = pyparse.Parser(4, 4)
        self.assertEqual('\t a([{b}])b"c\'d\n'.translate(pyparse.trans), 'xxx(((x)))x"x\'x\n')

class PyParseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.parser = pyparse.Parser(indentwidth=4, tabwidth=4)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        del cls.parser

    def test_init(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.parser.indentwidth, 4)
        self.assertEqual(self.parser.tabwidth, 4)

    def test_set_code(self):
        if False:
            return 10
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        with self.assertRaises(AssertionError):
            setcode('a')
        tests = ('', 'a\n')
        for string in tests:
            with self.subTest(string=string):
                setcode(string)
                eq(p.code, string)
                eq(p.study_level, 0)

    def test_find_good_parse_start(self):
        if False:
            i = 10
            return i + 15
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        start = p.find_good_parse_start

        def char_in_string_false(index):
            if False:
                i = 10
                return i + 15
            return False
        setcode('def spam():\n')
        eq(start(char_in_string_false), 0)
        setcode('class spam( ' + ' \n')
        eq(start(char_in_string_false), 0)
        setcode('"""This is a module docstring"""\nclass C:\n    def __init__(self, a,\n                 b=True):\n        pass\n')
        (pos0, pos) = (33, 42)
        with self.assertRaises(TypeError):
            start()
        with self.assertRaises(TypeError):
            start(False)
        self.assertIsNone(start(is_char_in_string=lambda index: True))
        eq(start(char_in_string_false), pos)
        eq(start(is_char_in_string=lambda index: index > pos), pos)
        eq(start(is_char_in_string=lambda index: index >= pos), pos0)
        eq(start(is_char_in_string=lambda index: index < pos), None)
        setcode('"""This is a module docstring"""\nclass C:\n    def __init__(self, a, b=True):\n        pass\n')
        eq(start(char_in_string_false), pos)
        eq(start(is_char_in_string=lambda index: index > pos), pos)
        eq(start(is_char_in_string=lambda index: index >= pos), pos0)
        eq(start(is_char_in_string=lambda index: index < pos), pos)

    def test_set_lo(self):
        if False:
            i = 10
            return i + 15
        code = '"""This is a module docstring"""\nclass C:\n    def __init__(self, a,\n                 b=True):\n        pass\n'
        pos = 42
        p = self.parser
        p.set_code(code)
        with self.assertRaises(AssertionError):
            p.set_lo(5)
        p.set_lo(0)
        self.assertEqual(p.code, code)
        p.set_lo(pos)
        self.assertEqual(p.code, code[pos:])

    def test_study1(self):
        if False:
            print('Hello World!')
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        study = p._study1
        (NONE, BACKSLASH, FIRST, NEXT, BRACKET) = range(5)
        TestInfo = namedtuple('TestInfo', ['string', 'goodlines', 'continuation'])
        tests = (TestInfo('', [0], NONE), TestInfo('"""This is a complete docstring."""\n', [0, 1], NONE), TestInfo("'''This is a complete docstring.'''\n", [0, 1], NONE), TestInfo('"""This is a continued docstring.\n', [0, 1], FIRST), TestInfo("'''This is a continued docstring.\n", [0, 1], FIRST), TestInfo('"""Closing quote does not match."\n', [0, 1], FIRST), TestInfo('"""Bracket in docstring [\n', [0, 1], FIRST), TestInfo("'''Incomplete two line docstring.\n\n", [0, 2], NEXT), TestInfo('"This is a complete string."\n', [0, 1], NONE), TestInfo('"This is an incomplete string.\n', [0, 1], NONE), TestInfo("'This is more incomplete.\n\n", [0, 1, 2], NONE), TestInfo('# Comment\\\n', [0, 1], NONE), TestInfo('("""Complete string in bracket"""\n', [0, 1], BRACKET), TestInfo('("""Open string in bracket\n', [0, 1], FIRST), TestInfo('a = (1 + 2) - 5 *\\\n', [0, 1], BACKSLASH), TestInfo('\n   def function1(self, a,\n                 b):\n', [0, 1, 3], NONE), TestInfo('\n   def function1(self, a,\\\n', [0, 1, 2], BRACKET), TestInfo('\n   def function1(self, a,\n', [0, 1, 2], BRACKET), TestInfo('())\n', [0, 1], NONE), TestInfo(')(\n', [0, 1], BRACKET), TestInfo('{)(]\n', [0, 1], NONE))
        for test in tests:
            with self.subTest(string=test.string):
                setcode(test.string)
                study()
                eq(p.study_level, 1)
                eq(p.goodlines, test.goodlines)
                eq(p.continuation, test.continuation)
        self.assertIsNone(study())

    def test_get_continuation_type(self):
        if False:
            while True:
                i = 10
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        gettype = p.get_continuation_type
        (NONE, BACKSLASH, FIRST, NEXT, BRACKET) = range(5)
        TestInfo = namedtuple('TestInfo', ['string', 'continuation'])
        tests = (TestInfo('', NONE), TestInfo('"""This is a continuation docstring.\n', FIRST), TestInfo("'''This is a multiline-continued docstring.\n\n", NEXT), TestInfo('a = (1 + 2) - 5 *\\\n', BACKSLASH), TestInfo('\n   def function1(self, a,\\\n', BRACKET))
        for test in tests:
            with self.subTest(string=test.string):
                setcode(test.string)
                eq(gettype(), test.continuation)

    def test_study2(self):
        if False:
            return 10
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        study = p._study2
        TestInfo = namedtuple('TestInfo', ['string', 'start', 'end', 'lastch', 'openbracket', 'bracketing'])
        tests = (TestInfo('', 0, 0, '', None, ((0, 0),)), TestInfo("'''This is a multiline continuation docstring.\n\n", 0, 48, "'", None, ((0, 0), (0, 1), (48, 0))), TestInfo(' # Comment\\\n', 0, 12, '', None, ((0, 0), (1, 1), (12, 0))), TestInfo(' #Comment\\\n', 0, 0, '', None, ((0, 0),)), TestInfo('a = (1 + 2) - 5 *\\\n', 0, 19, '*', None, ((0, 0), (4, 1), (11, 0))), TestInfo('\n   def function1(self, a,\n                 b):\n', 1, 48, ':', None, ((1, 0), (17, 1), (46, 0))), TestInfo('\n   def function1(self, a,\\\n', 1, 28, ',', 17, ((1, 0), (17, 1))), TestInfo('\n   def function1(self, a,\n', 1, 27, ',', 17, ((1, 0), (17, 1))), TestInfo('\n   def function1(self, a,  # End of line comment.\n', 1, 51, ',', 17, ((1, 0), (17, 1), (28, 2), (51, 1))), TestInfo('  a = ["first item",\n  # Comment line\n    "next item",\n', 0, 55, ',', 6, ((0, 0), (6, 1), (7, 2), (19, 1), (23, 2), (38, 1), (42, 2), (53, 1))), TestInfo('())\n', 0, 4, ')', None, ((0, 0), (0, 1), (2, 0), (3, 0))), TestInfo(')(\n', 0, 3, '(', 1, ((0, 0), (1, 0), (1, 1))), TestInfo('{)(]\n', 0, 5, ']', None, ((0, 0), (0, 1), (2, 0), (2, 1), (4, 0))), TestInfo(':\\a\n', 0, 4, '\\a', None, ((0, 0),)), TestInfo('\n', 0, 0, '', None, ((0, 0),)))
        for test in tests:
            with self.subTest(string=test.string):
                setcode(test.string)
                study()
                eq(p.study_level, 2)
                eq(p.stmt_start, test.start)
                eq(p.stmt_end, test.end)
                eq(p.lastch, test.lastch)
                eq(p.lastopenbracketpos, test.openbracket)
                eq(p.stmt_bracketing, test.bracketing)
        self.assertIsNone(study())

    def test_get_num_lines_in_stmt(self):
        if False:
            return 10
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        getlines = p.get_num_lines_in_stmt
        TestInfo = namedtuple('TestInfo', ['string', 'lines'])
        tests = (TestInfo('[x for x in a]\n', 1), TestInfo('[x\nfor x in a\n', 2), TestInfo('[x\\\nfor x in a\\\n', 2), TestInfo('[x\nfor x in a\n]\n', 3), TestInfo('\n"""Docstring comment L1"""\nL2\nL3\nL4\n', 1), TestInfo('\n"""Docstring comment L1\nL2"""\nL3\nL4\n', 1), TestInfo('\n"""Docstring comment L1\\\nL2\\\nL3\\\nL4\\\n', 4), TestInfo('\n\n"""Docstring comment L1\\\nL2\\\nL3\\\nL4\\\n"""\n', 5))
        setcode('')
        with self.assertRaises(IndexError):
            getlines()
        for test in tests:
            with self.subTest(string=test.string):
                setcode(test.string)
                eq(getlines(), test.lines)

    def test_compute_bracket_indent(self):
        if False:
            while True:
                i = 10
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        indent = p.compute_bracket_indent
        TestInfo = namedtuple('TestInfo', ['string', 'spaces'])
        tests = (TestInfo('def function1(self, a,\n', 14), TestInfo('\n    def function1(self, a,\n', 18), TestInfo('\n\tdef function1(self, a,\n', 18), TestInfo('\n    def function1(\n', 8), TestInfo('\n\tdef function1(\n', 8), TestInfo('\n    def function1(  \n', 8), TestInfo('[\n"first item",\n  # Comment line\n    "next item",\n', 0), TestInfo('[\n  "first item",\n  # Comment line\n    "next item",\n', 2), TestInfo('["first item",\n  # Comment line\n    "next item",\n', 1), TestInfo('(\n', 4), TestInfo('(a\n', 1))
        setcode('def function1(self, a, b):\n')
        with self.assertRaises(AssertionError):
            indent()
        for test in tests:
            setcode(test.string)
            eq(indent(), test.spaces)

    def test_compute_backslash_indent(self):
        if False:
            return 10
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        indent = p.compute_backslash_indent
        errors = ('def function1(self, a, b\\\n', '    """ (\\\n', 'a = #\\\n')
        for string in errors:
            with self.subTest(string=string):
                setcode(string)
                with self.assertRaises(AssertionError):
                    indent()
        TestInfo = namedtuple('TestInfo', ('string', 'spaces'))
        tests = (TestInfo('a = (1 + 2) - 5 *\\\n', 4), TestInfo('a = 1 + 2 - 5 *\\\n', 4), TestInfo('    a = 1 + 2 - 5 *\\\n', 8), TestInfo('  a = "spam"\\\n', 6), TestInfo('  a = \\\n"a"\\\n', 4), TestInfo('  a = #\\\n"a"\\\n', 5), TestInfo('a == \\\n', 2), TestInfo('a != \\\n', 2), TestInfo('\\\n', 2), TestInfo('    \\\n', 6), TestInfo('\t\\\n', 6), TestInfo('a\\\n', 3), TestInfo('{}\\\n', 4), TestInfo('(1 + 2) - 5 *\\\n', 3))
        for test in tests:
            with self.subTest(string=test.string):
                setcode(test.string)
                eq(indent(), test.spaces)

    def test_get_base_indent_string(self):
        if False:
            for i in range(10):
                print('nop')
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        baseindent = p.get_base_indent_string
        TestInfo = namedtuple('TestInfo', ['string', 'indent'])
        tests = (TestInfo('', ''), TestInfo('def a():\n', ''), TestInfo('\tdef a():\n', '\t'), TestInfo('    def a():\n', '    '), TestInfo('    def a(\n', '    '), TestInfo('\t\n    def a(\n', '    '), TestInfo('\t\n    # Comment.\n', '    '))
        for test in tests:
            with self.subTest(string=test.string):
                setcode(test.string)
                eq(baseindent(), test.indent)

    def test_is_block_opener(self):
        if False:
            print('Hello World!')
        yes = self.assertTrue
        no = self.assertFalse
        p = self.parser
        setcode = p.set_code
        opener = p.is_block_opener
        TestInfo = namedtuple('TestInfo', ['string', 'assert_'])
        tests = (TestInfo('def a():\n', yes), TestInfo('\n   def function1(self, a,\n                 b):\n', yes), TestInfo(':\n', yes), TestInfo('a:\n', yes), TestInfo('):\n', yes), TestInfo('(:\n', yes), TestInfo('":\n', no), TestInfo('\n   def function1(self, a,\n', no), TestInfo('def function1(self, a):\n    pass\n', no), TestInfo('# A comment:\n', no), TestInfo('"""A docstring:\n', no), TestInfo('"""A docstring:\n', no))
        for test in tests:
            with self.subTest(string=test.string):
                setcode(test.string)
                test.assert_(opener())

    def test_is_block_closer(self):
        if False:
            for i in range(10):
                print('nop')
        yes = self.assertTrue
        no = self.assertFalse
        p = self.parser
        setcode = p.set_code
        closer = p.is_block_closer
        TestInfo = namedtuple('TestInfo', ['string', 'assert_'])
        tests = (TestInfo('return\n', yes), TestInfo('\tbreak\n', yes), TestInfo('  continue\n', yes), TestInfo('     raise\n', yes), TestInfo('pass    \n', yes), TestInfo('pass\t\n', yes), TestInfo('return #\n', yes), TestInfo('raised\n', no), TestInfo('returning\n', no), TestInfo('# return\n', no), TestInfo('"""break\n', no), TestInfo('"continue\n', no), TestInfo('def function1(self, a):\n    pass\n', yes))
        for test in tests:
            with self.subTest(string=test.string):
                setcode(test.string)
                test.assert_(closer())

    def test_get_last_stmt_bracketing(self):
        if False:
            while True:
                i = 10
        eq = self.assertEqual
        p = self.parser
        setcode = p.set_code
        bracketing = p.get_last_stmt_bracketing
        TestInfo = namedtuple('TestInfo', ['string', 'bracket'])
        tests = (TestInfo('', ((0, 0),)), TestInfo('a\n', ((0, 0),)), TestInfo('()()\n', ((0, 0), (0, 1), (2, 0), (2, 1), (4, 0))), TestInfo('(\n)()\n', ((0, 0), (0, 1), (3, 0), (3, 1), (5, 0))), TestInfo('()\n()\n', ((3, 0), (3, 1), (5, 0))), TestInfo('()(\n)\n', ((0, 0), (0, 1), (2, 0), (2, 1), (5, 0))), TestInfo('(())\n', ((0, 0), (0, 1), (1, 2), (3, 1), (4, 0))), TestInfo('(\n())\n', ((0, 0), (0, 1), (2, 2), (4, 1), (5, 0))), TestInfo('{)(]\n', ((0, 0), (0, 1), (2, 0), (2, 1), (4, 0))), TestInfo('(((())\n', ((0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (5, 3), (6, 2))))
        for test in tests:
            with self.subTest(string=test.string):
                setcode(test.string)
                eq(bracketing(), test.bracket)
if __name__ == '__main__':
    unittest.main(verbosity=2)