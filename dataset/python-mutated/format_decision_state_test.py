"""Tests for yapf.format_decision_state."""
import textwrap
import unittest
from yapf.pytree import pytree_utils
from yapf.yapflib import format_decision_state
from yapf.yapflib import logical_line
from yapf.yapflib import style
from yapftests import yapf_test_helper

class FormatDecisionStateTest(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        style.SetGlobalStyle(style.CreateYapfStyle())

    def testSimpleFunctionDefWithNoSplitting(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        def f(a, b):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        lline = logical_line.LogicalLine(0, _FilterLine(llines[0]))
        lline.CalculateFormattingInformation()
        state = format_decision_state.FormatDecisionState(lline, 0)
        state.MoveStateToNextToken()
        self.assertEqual('f', state.next_token.value)
        self.assertFalse(state.CanSplit(False))
        state.AddTokenToState(False, True)
        self.assertEqual('(', state.next_token.value)
        self.assertFalse(state.CanSplit(False))
        self.assertFalse(state.MustSplit())
        state.AddTokenToState(False, True)
        self.assertEqual('a', state.next_token.value)
        self.assertTrue(state.CanSplit(False))
        self.assertFalse(state.MustSplit())
        state.AddTokenToState(False, True)
        self.assertEqual(',', state.next_token.value)
        self.assertFalse(state.CanSplit(False))
        self.assertFalse(state.MustSplit())
        state.AddTokenToState(False, True)
        self.assertEqual('b', state.next_token.value)
        self.assertTrue(state.CanSplit(False))
        self.assertFalse(state.MustSplit())
        state.AddTokenToState(False, True)
        self.assertEqual(')', state.next_token.value)
        self.assertTrue(state.CanSplit(False))
        self.assertFalse(state.MustSplit())
        state.AddTokenToState(False, True)
        self.assertEqual(':', state.next_token.value)
        self.assertFalse(state.CanSplit(False))
        self.assertFalse(state.MustSplit())
        clone = state.Clone()
        self.assertEqual(repr(state), repr(clone))

    def testSimpleFunctionDefWithSplitting(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        def f(a, b):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        lline = logical_line.LogicalLine(0, _FilterLine(llines[0]))
        lline.CalculateFormattingInformation()
        state = format_decision_state.FormatDecisionState(lline, 0)
        state.MoveStateToNextToken()
        self.assertEqual('f', state.next_token.value)
        self.assertFalse(state.CanSplit(False))
        state.AddTokenToState(True, True)
        self.assertEqual('(', state.next_token.value)
        self.assertFalse(state.CanSplit(False))
        state.AddTokenToState(True, True)
        self.assertEqual('a', state.next_token.value)
        self.assertTrue(state.CanSplit(False))
        state.AddTokenToState(True, True)
        self.assertEqual(',', state.next_token.value)
        self.assertFalse(state.CanSplit(False))
        state.AddTokenToState(True, True)
        self.assertEqual('b', state.next_token.value)
        self.assertTrue(state.CanSplit(False))
        state.AddTokenToState(True, True)
        self.assertEqual(')', state.next_token.value)
        self.assertTrue(state.CanSplit(False))
        state.AddTokenToState(True, True)
        self.assertEqual(':', state.next_token.value)
        self.assertFalse(state.CanSplit(False))
        clone = state.Clone()
        self.assertEqual(repr(state), repr(clone))

def _FilterLine(lline):
    if False:
        i = 10
        return i + 15
    'Filter out nonsemantic tokens from the LogicalLines.'
    return [ft for ft in lline.tokens if ft.name not in pytree_utils.NONSEMANTIC_TOKENS]
if __name__ == '__main__':
    unittest.main()