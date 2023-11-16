"""Tests for yapf.comment_splicer."""
import textwrap
import unittest
from yapf.pytree import comment_splicer
from yapf.pytree import pytree_utils
from yapftests import yapf_test_helper

class CommentSplicerTest(yapf_test_helper.YAPFTest):

    def _AssertNodeType(self, expected_type, node):
        if False:
            i = 10
            return i + 15
        self.assertEqual(expected_type, pytree_utils.NodeName(node))

    def _AssertNodeIsComment(self, node, text_in_comment=None):
        if False:
            print('Hello World!')
        if pytree_utils.NodeName(node) == 'simple_stmt':
            self._AssertNodeType('COMMENT', node.children[0])
            node_value = node.children[0].value
        else:
            self._AssertNodeType('COMMENT', node)
            node_value = node.value
        if text_in_comment is not None:
            self.assertIn(text_in_comment, node_value)

    def _FindNthChildNamed(self, node, name, n=1):
        if False:
            i = 10
            return i + 15
        for (i, child) in enumerate([c for c in node.pre_order() if pytree_utils.NodeName(c) == name]):
            if i == n - 1:
                return child
        raise RuntimeError('No Nth child for n={0}'.format(n))

    def testSimpleInline(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        foo = 1 # and a comment\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        expr = tree.children[0].children[0]
        self._AssertNodeType('expr_stmt', expr)
        self.assertEqual(4, len(expr.children))
        comment_node = expr.children[3]
        self._AssertNodeIsComment(comment_node, '# and a comment')

    def testSimpleSeparateLine(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        foo = 1\n        # first comment\n        bar = 2\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        self.assertEqual(4, len(tree.children))
        comment_node = tree.children[1]
        self._AssertNodeIsComment(comment_node)

    def testTwoLineComment(self):
        if False:
            return 10
        code = textwrap.dedent('        foo = 1\n        # first comment\n        # second comment\n        bar = 2\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        self.assertEqual(4, len(tree.children))
        self._AssertNodeIsComment(tree.children[1])

    def testCommentIsFirstChildInCompound(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('\n        if x:\n          # a comment\n          foo = 1\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        if_suite = tree.children[0].children[3]
        self._AssertNodeType('NEWLINE', if_suite.children[0])
        self._AssertNodeIsComment(if_suite.children[1])

    def testCommentIsLastChildInCompound(self):
        if False:
            return 10
        code = textwrap.dedent('        if x:\n          foo = 1\n          # a comment\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        if_suite = tree.children[0].children[3]
        self._AssertNodeType('DEDENT', if_suite.children[-1])
        self._AssertNodeIsComment(if_suite.children[-2])

    def testInlineAfterSeparateLine(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        bar = 1\n        # line comment\n        foo = 1 # inline comment\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        sep_comment_node = tree.children[1]
        self._AssertNodeIsComment(sep_comment_node, '# line comment')
        expr = tree.children[2].children[0]
        inline_comment_node = expr.children[-1]
        self._AssertNodeIsComment(inline_comment_node, '# inline comment')

    def testSeparateLineAfterInline(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        bar = 1\n        foo = 1 # inline comment\n        # line comment\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        sep_comment_node = tree.children[-2]
        self._AssertNodeIsComment(sep_comment_node, '# line comment')
        expr = tree.children[1].children[0]
        inline_comment_node = expr.children[-1]
        self._AssertNodeIsComment(inline_comment_node, '# inline comment')

    def testCommentBeforeDedent(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        if bar:\n          z = 1\n        # a comment\n        j = 2\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        self._AssertNodeIsComment(tree.children[1])
        if_suite = tree.children[0].children[3]
        self._AssertNodeType('DEDENT', if_suite.children[-1])

    def testCommentBeforeDedentTwoLevel(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        if foo:\n          if bar:\n            z = 1\n          # a comment\n        y = 1\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        if_suite = tree.children[0].children[3]
        self._AssertNodeIsComment(if_suite.children[-2])
        self._AssertNodeType('DEDENT', if_suite.children[-1])

    def testCommentBeforeDedentTwoLevelImproperlyIndented(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        if foo:\n          if bar:\n            z = 1\n           # comment 2\n        y = 1\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        if_suite = tree.children[0].children[3]
        self._AssertNodeIsComment(if_suite.children[-2])
        self._AssertNodeType('DEDENT', if_suite.children[-1])

    def testCommentBeforeDedentThreeLevel(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        if foo:\n          if bar:\n            z = 1\n            # comment 2\n          # comment 1\n        # comment 0\n        j = 2\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        self._AssertNodeIsComment(tree.children[1], '# comment 0')
        if_suite_1 = self._FindNthChildNamed(tree, 'suite', n=1)
        self._AssertNodeIsComment(if_suite_1.children[-2], '# comment 1')
        self._AssertNodeType('DEDENT', if_suite_1.children[-1])
        if_suite_2 = self._FindNthChildNamed(tree, 'suite', n=2)
        self._AssertNodeIsComment(if_suite_2.children[-2], '# comment 2')
        self._AssertNodeType('DEDENT', if_suite_2.children[-1])

    def testCommentsInClass(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        class Foo:\n          '''docstring abc...'''\n          # top-level comment\n          def foo(): pass\n          # another comment\n    ")
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        class_suite = tree.children[0].children[3]
        another_comment = class_suite.children[-2]
        self._AssertNodeIsComment(another_comment, '# another')
        funcdef = class_suite.children[3]
        toplevel_comment = funcdef.children[0]
        self._AssertNodeIsComment(toplevel_comment, '# top-level')

    def testMultipleBlockComments(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        # Block comment number 1\n\n        # Block comment number 2\n        def f():\n          pass\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        funcdef = tree.children[0]
        block_comment_1 = funcdef.children[0]
        self._AssertNodeIsComment(block_comment_1, '# Block comment number 1')
        block_comment_2 = funcdef.children[1]
        self._AssertNodeIsComment(block_comment_2, '# Block comment number 2')

    def testCommentsOnDedents(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        class Foo(object):\n          # A comment for qux.\n          def qux(self):\n            pass\n\n          # Interim comment.\n\n          def mux(self):\n            pass\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        classdef = tree.children[0]
        class_suite = classdef.children[6]
        qux_comment = class_suite.children[1]
        self._AssertNodeIsComment(qux_comment, '# A comment for qux.')
        interim_comment = class_suite.children[4]
        self._AssertNodeIsComment(interim_comment, '# Interim comment.')

    def testExprComments(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        foo( # Request fractions of an hour.\n          948.0/3600, 20)\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        trailer = self._FindNthChildNamed(tree, 'trailer', 1)
        comment = trailer.children[1]
        self._AssertNodeIsComment(comment, '# Request fractions of an hour.')

    def testMultipleCommentsInOneExpr(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        foo( # com 1\n          948.0/3600, # com 2\n          20 + 12 # com 3\n          )\n    ')
        tree = pytree_utils.ParseCodeToTree(code)
        comment_splicer.SpliceComments(tree)
        trailer = self._FindNthChildNamed(tree, 'trailer', 1)
        self._AssertNodeIsComment(trailer.children[1], '# com 1')
        arglist = self._FindNthChildNamed(tree, 'arglist', 1)
        self._AssertNodeIsComment(arglist.children[2], '# com 2')
        arith_expr = self._FindNthChildNamed(tree, 'arith_expr', 1)
        self._AssertNodeIsComment(arith_expr.children[-1], '# com 3')
if __name__ == '__main__':
    unittest.main()