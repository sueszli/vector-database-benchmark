"""Tests for yapf.line_joiner."""
import textwrap
import unittest
from yapf.yapflib import line_joiner
from yapf.yapflib import style
from yapftests import yapf_test_helper

class LineJoinerTest(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        style.SetGlobalStyle(style.CreatePEP8Style())

    def _CheckLineJoining(self, code, join_lines):
        if False:
            print('Hello World!')
        'Check that the given LogicalLines are joined as expected.\n\n    Arguments:\n      code: The code to check to see if we can join it.\n      join_lines: True if we expect the lines to be joined.\n    '
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(line_joiner.CanMergeMultipleLines(llines), join_lines)

    def testSimpleSingleLineStatement(self):
        if False:
            return 10
        code = textwrap.dedent('        if isinstance(a, int): continue\n    ')
        self._CheckLineJoining(code, join_lines=True)

    def testSimpleMultipleLineStatement(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        if isinstance(b, int):\n            continue\n    ')
        self._CheckLineJoining(code, join_lines=False)

    def testSimpleMultipleLineComplexStatement(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        if isinstance(c, int):\n            while True:\n                continue\n    ')
        self._CheckLineJoining(code, join_lines=False)

    def testSimpleMultipleLineStatementWithComment(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent("        if isinstance(d, int): continue  # We're pleased that d's an int.\n    ")
        self._CheckLineJoining(code, join_lines=True)

    def testSimpleMultipleLineStatementWithLargeIndent(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        if isinstance(e, int):    continue\n    ')
        self._CheckLineJoining(code, join_lines=True)

    def testOverColumnLimit(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        if instance(bbbbbbbbbbbbbbbbbbbbbbbbb, int): cccccccccccccccccccccccccc = ddddddddddddddddddddd\n    ')
        self._CheckLineJoining(code, join_lines=False)
if __name__ == '__main__':
    unittest.main()