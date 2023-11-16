"""Tests for yapf.subtype_assigner."""
import textwrap
import unittest
from yapf.pytree import pytree_utils
from yapf.yapflib import format_token
from yapf.yapflib import subtypes
from yapftests import yapf_test_helper

class SubtypeAssignerTest(yapf_test_helper.YAPFTest):

    def _CheckFormatTokenSubtypes(self, llines, list_of_expected):
        if False:
            for i in range(10):
                print('nop')
        'Check that the tokens in the LogicalLines have the expected subtypes.\n\n    Args:\n      llines: list of LogicalLine.\n      list_of_expected: list of (name, subtype) pairs. Non-semantic tokens are\n        filtered out from the expected values.\n    '
        actual = []
        for lline in llines:
            filtered_values = [(ft.value, ft.subtypes) for ft in lline.tokens if ft.name not in pytree_utils.NONSEMANTIC_TOKENS]
            if filtered_values:
                actual.append(filtered_values)
        self.assertEqual(list_of_expected, actual)

    def testFuncDefDefaultAssign(self):
        if False:
            return 10
        self.maxDiff = None
        code = textwrap.dedent('        def foo(a=37, *b, **c):\n          return -x[:42]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('def', {subtypes.NONE}), ('foo', {subtypes.FUNC_DEF}), ('(', {subtypes.NONE}), ('a', {subtypes.NONE, subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST, subtypes.PARAMETER_START}), ('=', {subtypes.DEFAULT_OR_NAMED_ASSIGN, subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST}), ('37', {subtypes.NONE, subtypes.PARAMETER_STOP, subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST}), (',', {subtypes.NONE}), ('*', {subtypes.PARAMETER_START, subtypes.VARARGS_STAR, subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST}), ('b', {subtypes.NONE, subtypes.PARAMETER_STOP, subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST}), (',', {subtypes.NONE}), ('**', {subtypes.PARAMETER_START, subtypes.KWARGS_STAR_STAR, subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST}), ('c', {subtypes.NONE, subtypes.PARAMETER_STOP, subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST}), (')', {subtypes.NONE}), (':', {subtypes.NONE})], [('return', {subtypes.NONE}), ('-', {subtypes.UNARY_OPERATOR}), ('x', {subtypes.NONE}), ('[', {subtypes.SUBSCRIPT_BRACKET}), (':', {subtypes.SUBSCRIPT_COLON}), ('42', {subtypes.NONE}), (']', {subtypes.SUBSCRIPT_BRACKET})]])

    def testFuncCallWithDefaultAssign(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        foo(x, a='hello world')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('foo', {subtypes.NONE}), ('(', {subtypes.NONE}), ('x', {subtypes.NONE, subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST}), (',', {subtypes.NONE}), ('a', {subtypes.NONE, subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST}), ('=', {subtypes.DEFAULT_OR_NAMED_ASSIGN}), ("'hello world'", {subtypes.NONE}), (')', {subtypes.NONE})]])

    def testSetComprehension(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        def foo(value):\n          return {value.lower()}\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('def', {subtypes.NONE}), ('foo', {subtypes.FUNC_DEF}), ('(', {subtypes.NONE}), ('value', {subtypes.NONE, subtypes.PARAMETER_START, subtypes.PARAMETER_STOP}), (')', {subtypes.NONE}), (':', {subtypes.NONE})], [('return', {subtypes.NONE}), ('{', {subtypes.NONE}), ('value', {subtypes.NONE}), ('.', {subtypes.NONE}), ('lower', {subtypes.NONE}), ('(', {subtypes.NONE}), (')', {subtypes.NONE}), ('}', {subtypes.NONE})]])
        code = textwrap.dedent('        def foo(strs):\n          return {s.lower() for s in strs}\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('def', {subtypes.NONE}), ('foo', {subtypes.FUNC_DEF}), ('(', {subtypes.NONE}), ('strs', {subtypes.NONE, subtypes.PARAMETER_START, subtypes.PARAMETER_STOP}), (')', {subtypes.NONE}), (':', {subtypes.NONE})], [('return', {subtypes.NONE}), ('{', {subtypes.NONE}), ('s', {subtypes.COMP_EXPR}), ('.', {subtypes.COMP_EXPR}), ('lower', {subtypes.COMP_EXPR}), ('(', {subtypes.COMP_EXPR}), (')', {subtypes.COMP_EXPR}), ('for', {subtypes.DICT_SET_GENERATOR, subtypes.COMP_FOR}), ('s', {subtypes.COMP_FOR}), ('in', {subtypes.COMP_FOR}), ('strs', {subtypes.COMP_FOR}), ('}', {subtypes.NONE})]])
        code = textwrap.dedent('        def foo(strs):\n          return {s + s.lower() for s in strs}\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('def', {subtypes.NONE}), ('foo', {subtypes.FUNC_DEF}), ('(', {subtypes.NONE}), ('strs', {subtypes.NONE, subtypes.PARAMETER_START, subtypes.PARAMETER_STOP}), (')', {subtypes.NONE}), (':', {subtypes.NONE})], [('return', {subtypes.NONE}), ('{', {subtypes.NONE}), ('s', {subtypes.COMP_EXPR}), ('+', {subtypes.BINARY_OPERATOR, subtypes.COMP_EXPR}), ('s', {subtypes.COMP_EXPR}), ('.', {subtypes.COMP_EXPR}), ('lower', {subtypes.COMP_EXPR}), ('(', {subtypes.COMP_EXPR}), (')', {subtypes.COMP_EXPR}), ('for', {subtypes.DICT_SET_GENERATOR, subtypes.COMP_FOR}), ('s', {subtypes.COMP_FOR}), ('in', {subtypes.COMP_FOR}), ('strs', {subtypes.COMP_FOR}), ('}', {subtypes.NONE})]])
        code = textwrap.dedent('        def foo(strs):\n          return {c.lower() for s in strs for c in s}\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('def', {subtypes.NONE}), ('foo', {subtypes.FUNC_DEF}), ('(', {subtypes.NONE}), ('strs', {subtypes.NONE, subtypes.PARAMETER_START, subtypes.PARAMETER_STOP}), (')', {subtypes.NONE}), (':', {subtypes.NONE})], [('return', {subtypes.NONE}), ('{', {subtypes.NONE}), ('c', {subtypes.COMP_EXPR}), ('.', {subtypes.COMP_EXPR}), ('lower', {subtypes.COMP_EXPR}), ('(', {subtypes.COMP_EXPR}), (')', {subtypes.COMP_EXPR}), ('for', {subtypes.DICT_SET_GENERATOR, subtypes.COMP_FOR, subtypes.COMP_EXPR}), ('s', {subtypes.COMP_FOR, subtypes.COMP_EXPR}), ('in', {subtypes.COMP_FOR, subtypes.COMP_EXPR}), ('strs', {subtypes.COMP_FOR, subtypes.COMP_EXPR}), ('for', {subtypes.DICT_SET_GENERATOR, subtypes.COMP_FOR}), ('c', {subtypes.COMP_FOR}), ('in', {subtypes.COMP_FOR}), ('s', {subtypes.COMP_FOR}), ('}', {subtypes.NONE})]])

    def testDictComprehension(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        def foo(value):\n          return {value: value.lower()}\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('def', {subtypes.NONE}), ('foo', {subtypes.FUNC_DEF}), ('(', {subtypes.NONE}), ('value', {subtypes.NONE, subtypes.PARAMETER_START, subtypes.PARAMETER_STOP}), (')', {subtypes.NONE}), (':', {subtypes.NONE})], [('return', {subtypes.NONE}), ('{', {subtypes.NONE}), ('value', {subtypes.DICTIONARY_KEY, subtypes.DICTIONARY_KEY_PART}), (':', {subtypes.NONE}), ('value', {subtypes.DICTIONARY_VALUE}), ('.', {subtypes.NONE}), ('lower', {subtypes.NONE}), ('(', {subtypes.NONE}), (')', {subtypes.NONE}), ('}', {subtypes.NONE})]])
        code = textwrap.dedent('        def foo(strs):\n          return {s: s.lower() for s in strs}\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('def', {subtypes.NONE}), ('foo', {subtypes.FUNC_DEF}), ('(', {subtypes.NONE}), ('strs', {subtypes.NONE, subtypes.PARAMETER_START, subtypes.PARAMETER_STOP}), (')', {subtypes.NONE}), (':', {subtypes.NONE})], [('return', {subtypes.NONE}), ('{', {subtypes.NONE}), ('s', {subtypes.DICTIONARY_KEY, subtypes.DICTIONARY_KEY_PART, subtypes.COMP_EXPR}), (':', {subtypes.COMP_EXPR}), ('s', {subtypes.DICTIONARY_VALUE, subtypes.COMP_EXPR}), ('.', {subtypes.COMP_EXPR}), ('lower', {subtypes.COMP_EXPR}), ('(', {subtypes.COMP_EXPR}), (')', {subtypes.COMP_EXPR}), ('for', {subtypes.DICT_SET_GENERATOR, subtypes.COMP_FOR}), ('s', {subtypes.COMP_FOR}), ('in', {subtypes.COMP_FOR}), ('strs', {subtypes.COMP_FOR}), ('}', {subtypes.NONE})]])
        code = textwrap.dedent('        def foo(strs):\n          return {c: c.lower() for s in strs for c in s}\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('def', {subtypes.NONE}), ('foo', {subtypes.FUNC_DEF}), ('(', {subtypes.NONE}), ('strs', {subtypes.NONE, subtypes.PARAMETER_START, subtypes.PARAMETER_STOP}), (')', {subtypes.NONE}), (':', {subtypes.NONE})], [('return', {subtypes.NONE}), ('{', {subtypes.NONE}), ('c', {subtypes.DICTIONARY_KEY, subtypes.DICTIONARY_KEY_PART, subtypes.COMP_EXPR}), (':', {subtypes.COMP_EXPR}), ('c', {subtypes.DICTIONARY_VALUE, subtypes.COMP_EXPR}), ('.', {subtypes.COMP_EXPR}), ('lower', {subtypes.COMP_EXPR}), ('(', {subtypes.COMP_EXPR}), (')', {subtypes.COMP_EXPR}), ('for', {subtypes.DICT_SET_GENERATOR, subtypes.COMP_FOR, subtypes.COMP_EXPR}), ('s', {subtypes.COMP_FOR, subtypes.COMP_EXPR}), ('in', {subtypes.COMP_FOR, subtypes.COMP_EXPR}), ('strs', {subtypes.COMP_FOR, subtypes.COMP_EXPR}), ('for', {subtypes.DICT_SET_GENERATOR, subtypes.COMP_FOR}), ('c', {subtypes.COMP_FOR}), ('in', {subtypes.COMP_FOR}), ('s', {subtypes.COMP_FOR}), ('}', {subtypes.NONE})]])

    def testUnaryNotOperator(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        not a\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('not', {subtypes.UNARY_OPERATOR}), ('a', {subtypes.NONE})]])

    def testBitwiseOperators(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        x = ((a | (b ^ 3) & c) << 3) >> 1\n        ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('x', {subtypes.NONE}), ('=', {subtypes.ASSIGN_OPERATOR}), ('(', {subtypes.NONE}), ('(', {subtypes.NONE}), ('a', {subtypes.NONE}), ('|', {subtypes.BINARY_OPERATOR}), ('(', {subtypes.NONE}), ('b', {subtypes.NONE}), ('^', {subtypes.BINARY_OPERATOR}), ('3', {subtypes.NONE}), (')', {subtypes.NONE}), ('&', {subtypes.BINARY_OPERATOR}), ('c', {subtypes.NONE}), (')', {subtypes.NONE}), ('<<', {subtypes.BINARY_OPERATOR}), ('3', {subtypes.NONE}), (')', {subtypes.NONE}), ('>>', {subtypes.BINARY_OPERATOR}), ('1', {subtypes.NONE})]])

    def testArithmeticOperators(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        x = ((a + (b - 3) * (1 % c) @ d) / 3) // 1\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('x', {subtypes.NONE}), ('=', {subtypes.ASSIGN_OPERATOR}), ('(', {subtypes.NONE}), ('(', {subtypes.NONE}), ('a', {subtypes.NONE}), ('+', {subtypes.BINARY_OPERATOR}), ('(', {subtypes.NONE}), ('b', {subtypes.NONE}), ('-', {subtypes.BINARY_OPERATOR, subtypes.SIMPLE_EXPRESSION}), ('3', {subtypes.NONE}), (')', {subtypes.NONE}), ('*', {subtypes.BINARY_OPERATOR}), ('(', {subtypes.NONE}), ('1', {subtypes.NONE}), ('%', {subtypes.BINARY_OPERATOR, subtypes.SIMPLE_EXPRESSION}), ('c', {subtypes.NONE}), (')', {subtypes.NONE}), ('@', {subtypes.BINARY_OPERATOR}), ('d', {subtypes.NONE}), (')', {subtypes.NONE}), ('/', {subtypes.BINARY_OPERATOR}), ('3', {subtypes.NONE}), (')', {subtypes.NONE}), ('//', {subtypes.BINARY_OPERATOR}), ('1', {subtypes.NONE})]])

    def testSubscriptColon(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        x[0:42:1]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('x', {subtypes.NONE}), ('[', {subtypes.SUBSCRIPT_BRACKET}), ('0', {subtypes.NONE}), (':', {subtypes.SUBSCRIPT_COLON}), ('42', {subtypes.NONE}), (':', {subtypes.SUBSCRIPT_COLON}), ('1', {subtypes.NONE}), (']', {subtypes.SUBSCRIPT_BRACKET})]])

    def testFunctionCallWithStarExpression(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        [a, *b]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self._CheckFormatTokenSubtypes(llines, [[('[', {subtypes.NONE}), ('a', {subtypes.NONE}), (',', {subtypes.NONE}), ('*', {subtypes.UNARY_OPERATOR, subtypes.VARARGS_STAR}), ('b', {subtypes.NONE}), (']', {subtypes.NONE})]])
if __name__ == '__main__':
    unittest.main()