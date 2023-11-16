"""Tests for yapf.logical_line."""
import textwrap
import unittest
from yapf_third_party._ylib2to3 import pytree
from yapf_third_party._ylib2to3.pgen2 import token
from yapf.pytree import split_penalty
from yapf.yapflib import format_token
from yapf.yapflib import logical_line
from yapftests import yapf_test_helper

class LogicalLineBasicTest(yapf_test_helper.YAPFTest):

    def testConstruction(self):
        if False:
            i = 10
            return i + 15
        toks = _MakeFormatTokenList([(token.DOT, '.', 'DOT'), (token.VBAR, '|', 'VBAR')])
        lline = logical_line.LogicalLine(20, toks)
        self.assertEqual(20, lline.depth)
        self.assertEqual(['DOT', 'VBAR'], [tok.name for tok in lline.tokens])

    def testFirstLast(self):
        if False:
            while True:
                i = 10
        toks = _MakeFormatTokenList([(token.DOT, '.', 'DOT'), (token.LPAR, '(', 'LPAR'), (token.VBAR, '|', 'VBAR')])
        lline = logical_line.LogicalLine(20, toks)
        self.assertEqual(20, lline.depth)
        self.assertEqual('DOT', lline.first.name)
        self.assertEqual('VBAR', lline.last.name)

    def testAsCode(self):
        if False:
            print('Hello World!')
        toks = _MakeFormatTokenList([(token.DOT, '.', 'DOT'), (token.LPAR, '(', 'LPAR'), (token.VBAR, '|', 'VBAR')])
        lline = logical_line.LogicalLine(2, toks)
        self.assertEqual('    . ( |', lline.AsCode())

    def testAppendToken(self):
        if False:
            for i in range(10):
                print('nop')
        lline = logical_line.LogicalLine(0)
        lline.AppendToken(_MakeFormatTokenLeaf(token.LPAR, '(', 'LPAR'))
        lline.AppendToken(_MakeFormatTokenLeaf(token.RPAR, ')', 'RPAR'))
        self.assertEqual(['LPAR', 'RPAR'], [tok.name for tok in lline.tokens])

class LogicalLineFormattingInformationTest(yapf_test_helper.YAPFTest):

    def testFuncDef(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        def f(a, b):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        f = llines[0].tokens[1]
        self.assertFalse(f.can_break_before)
        self.assertFalse(f.must_break_before)
        self.assertEqual(f.split_penalty, split_penalty.UNBREAKABLE)
        lparen = llines[0].tokens[2]
        self.assertFalse(lparen.can_break_before)
        self.assertFalse(lparen.must_break_before)
        self.assertEqual(lparen.split_penalty, split_penalty.UNBREAKABLE)

def _MakeFormatTokenLeaf(token_type, token_value, name):
    if False:
        while True:
            i = 10
    return format_token.FormatToken(pytree.Leaf(token_type, token_value), name)

def _MakeFormatTokenList(token_type_values):
    if False:
        i = 10
        return i + 15
    return [_MakeFormatTokenLeaf(token_type, token_value, token_name) for (token_type, token_value, token_name) in token_type_values]
if __name__ == '__main__':
    unittest.main()