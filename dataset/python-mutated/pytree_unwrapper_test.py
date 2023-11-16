"""Tests for yapf.pytree_unwrapper."""
import textwrap
import unittest
from yapf.pytree import pytree_utils
from yapftests import yapf_test_helper

class PytreeUnwrapperTest(yapf_test_helper.YAPFTest):

    def _CheckLogicalLines(self, llines, list_of_expected):
        if False:
            for i in range(10):
                print('nop')
        'Check that the given LogicalLines match expectations.\n\n    Args:\n      llines: list of LogicalLine\n      list_of_expected: list of (depth, values) pairs. Non-semantic tokens are\n        filtered out from the expected values.\n    '
        actual = []
        for lline in llines:
            filtered_values = [ft.value for ft in lline.tokens if ft.name not in pytree_utils.NONSEMANTIC_TOKENS]
            actual.append((lline.depth, filtered_values))
        self.assertEqual(list_of_expected, actual)

    def testSimpleFileScope(self):
        if False:
            return 10
        code = textwrap.dedent('        x = 1\n        # a comment\n        y = 2\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['x', '=', '1']), (0, ['# a comment']), (0, ['y', '=', '2'])])

    def testSimpleMultilineStatement(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        y = (1 +\n             x)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['y', '=', '(', '1', '+', 'x', ')'])])

    def testFileScopeWithInlineComment(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        x = 1    # a comment\n        y = 2\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['x', '=', '1', '# a comment']), (0, ['y', '=', '2'])])

    def testSimpleIf(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        if foo:\n            x = 1\n            y = 2\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['if', 'foo', ':']), (1, ['x', '=', '1']), (1, ['y', '=', '2'])])

    def testSimpleIfWithComments(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        # c1\n        if foo: # c2\n            x = 1\n            y = 2\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['# c1']), (0, ['if', 'foo', ':', '# c2']), (1, ['x', '=', '1']), (1, ['y', '=', '2'])])

    def testIfWithCommentsInside(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        if foo:\n            # c1\n            x = 1 # c2\n            # c3\n            y = 2\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['if', 'foo', ':']), (1, ['# c1']), (1, ['x', '=', '1', '# c2']), (1, ['# c3']), (1, ['y', '=', '2'])])

    def testIfElifElse(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        if x:\n          x = 1 # c1\n        elif y: # c2\n          y = 1\n        else:\n          # c3\n          z = 1\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['if', 'x', ':']), (1, ['x', '=', '1', '# c1']), (0, ['elif', 'y', ':', '# c2']), (1, ['y', '=', '1']), (0, ['else', ':']), (1, ['# c3']), (1, ['z', '=', '1'])])

    def testNestedCompoundTwoLevel(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        if x:\n          x = 1 # c1\n          while t:\n            # c2\n            j = 1\n          k = 1\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['if', 'x', ':']), (1, ['x', '=', '1', '# c1']), (1, ['while', 't', ':']), (2, ['# c2']), (2, ['j', '=', '1']), (1, ['k', '=', '1'])])

    def testSimpleWhile(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        while x > 1: # c1\n           # c2\n           x = 1\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['while', 'x', '>', '1', ':', '# c1']), (1, ['# c2']), (1, ['x', '=', '1'])])

    def testSimpleTry(self):
        if False:
            return 10
        code = textwrap.dedent('        try:\n          pass\n        except:\n          pass\n        except:\n          pass\n        else:\n          pass\n        finally:\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['try', ':']), (1, ['pass']), (0, ['except', ':']), (1, ['pass']), (0, ['except', ':']), (1, ['pass']), (0, ['else', ':']), (1, ['pass']), (0, ['finally', ':']), (1, ['pass'])])

    def testSimpleFuncdef(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        def foo(x): # c1\n          # c2\n          return x\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['def', 'foo', '(', 'x', ')', ':', '# c1']), (1, ['# c2']), (1, ['return', 'x'])])

    def testTwoFuncDefs(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        def foo(x): # c1\n          # c2\n          return x\n\n        def bar(): # c3\n          # c4\n          return x\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['def', 'foo', '(', 'x', ')', ':', '# c1']), (1, ['# c2']), (1, ['return', 'x']), (0, ['def', 'bar', '(', ')', ':', '# c3']), (1, ['# c4']), (1, ['return', 'x'])])

    def testSimpleClassDef(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        class Klass: # c1\n          # c2\n          p = 1\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['class', 'Klass', ':', '# c1']), (1, ['# c2']), (1, ['p', '=', '1'])])

    def testSingleLineStmtInFunc(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        def f(): return 37\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['def', 'f', '(', ')', ':']), (1, ['return', '37'])])

    def testMultipleComments(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        # Comment #1\n\n        # Comment #2\n        def f():\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['# Comment #1']), (0, ['# Comment #2']), (0, ['def', 'f', '(', ')', ':']), (1, ['pass'])])

    def testSplitListWithComment(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent("        a = [\n            'a',\n            'b',\n            'c',  # hello world\n        ]\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckLogicalLines(llines, [(0, ['a', '=', '[', "'a'", ',', "'b'", ',', "'c'", ',', '# hello world', ']'])])

class MatchBracketsTest(yapf_test_helper.YAPFTest):

    def _CheckMatchingBrackets(self, llines, list_of_expected):
        if False:
            return 10
        'Check that the tokens have the expected matching bracket.\n\n    Arguments:\n      llines: list of LogicalLine.\n      list_of_expected: list of (index, index) pairs. The matching brackets at\n        the indexes need to match. Non-semantic tokens are filtered out from the\n        expected values.\n    '
        actual = []
        for lline in llines:
            filtered_values = [(ft, ft.matching_bracket) for ft in lline.tokens if ft.name not in pytree_utils.NONSEMANTIC_TOKENS]
            if filtered_values:
                actual.append(filtered_values)
        for (index, bracket_list) in enumerate(list_of_expected):
            lline = actual[index]
            if not bracket_list:
                for value in lline:
                    self.assertIsNone(value[1])
            else:
                for (open_bracket, close_bracket) in bracket_list:
                    self.assertEqual(lline[open_bracket][0], lline[close_bracket][1])
                    self.assertEqual(lline[close_bracket][0], lline[open_bracket][1])

    def testFunctionDef(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        def foo(a, b=['w','d'], c=[42, 37]):\n          pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckMatchingBrackets(llines, [[(2, 20), (7, 11), (15, 19)], []])

    def testDecorator(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        @bar()\n        def foo(a, b, c):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckMatchingBrackets(llines, [[(2, 3)], [(2, 8)], []])

    def testClassDef(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        class A(B, C, D):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckMatchingBrackets(llines, [[(2, 8)], []])
if __name__ == '__main__':
    unittest.main()