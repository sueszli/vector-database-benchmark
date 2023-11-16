"""Tests for yapf.format_token."""
import unittest
from yapf_third_party._ylib2to3 import pytree
from yapf_third_party._ylib2to3.pgen2 import token
from yapf.yapflib import format_token
from yapftests import yapf_test_helper

class TabbedContinuationAlignPaddingTest(yapf_test_helper.YAPFTest):

    def testSpace(self):
        if False:
            while True:
                i = 10
        align_style = 'SPACE'
        pad = format_token._TabbedContinuationAlignPadding(0, align_style, 2)
        self.assertEqual(pad, '')
        pad = format_token._TabbedContinuationAlignPadding(2, align_style, 2)
        self.assertEqual(pad, ' ' * 2)
        pad = format_token._TabbedContinuationAlignPadding(5, align_style, 2)
        self.assertEqual(pad, ' ' * 5)

    def testFixed(self):
        if False:
            while True:
                i = 10
        align_style = 'FIXED'
        pad = format_token._TabbedContinuationAlignPadding(0, align_style, 4)
        self.assertEqual(pad, '')
        pad = format_token._TabbedContinuationAlignPadding(2, align_style, 4)
        self.assertEqual(pad, '\t')
        pad = format_token._TabbedContinuationAlignPadding(5, align_style, 4)
        self.assertEqual(pad, '\t' * 2)

    def testVAlignRight(self):
        if False:
            for i in range(10):
                print('nop')
        align_style = 'VALIGN-RIGHT'
        pad = format_token._TabbedContinuationAlignPadding(0, align_style, 4)
        self.assertEqual(pad, '')
        pad = format_token._TabbedContinuationAlignPadding(2, align_style, 4)
        self.assertEqual(pad, '\t')
        pad = format_token._TabbedContinuationAlignPadding(4, align_style, 4)
        self.assertEqual(pad, '\t')
        pad = format_token._TabbedContinuationAlignPadding(5, align_style, 4)
        self.assertEqual(pad, '\t' * 2)

class FormatTokenTest(yapf_test_helper.YAPFTest):

    def testSimple(self):
        if False:
            while True:
                i = 10
        tok = format_token.FormatToken(pytree.Leaf(token.STRING, "'hello world'"), 'STRING')
        self.assertEqual("FormatToken(name=DOCSTRING, value='hello world', column=0, lineno=0, splitpenalty=0)", str(tok))
        self.assertTrue(tok.is_string)
        tok = format_token.FormatToken(pytree.Leaf(token.COMMENT, '# A comment'), 'COMMENT')
        self.assertEqual('FormatToken(name=COMMENT, value=# A comment, column=0, lineno=0, splitpenalty=0)', str(tok))
        self.assertTrue(tok.is_comment)

    def testIsMultilineString(self):
        if False:
            for i in range(10):
                print('nop')
        tok = format_token.FormatToken(pytree.Leaf(token.STRING, '"""hello"""'), 'STRING')
        self.assertTrue(tok.is_multiline_string)
        tok = format_token.FormatToken(pytree.Leaf(token.STRING, 'r"""hello"""'), 'STRING')
        self.assertTrue(tok.is_multiline_string)
if __name__ == '__main__':
    unittest.main()