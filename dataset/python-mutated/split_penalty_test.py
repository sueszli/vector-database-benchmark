"""Tests for yapf.split_penalty."""
import sys
import textwrap
import unittest
from yapf_third_party._ylib2to3 import pytree
from yapf.pytree import pytree_utils
from yapf.pytree import pytree_visitor
from yapf.pytree import split_penalty
from yapf.yapflib import style
from yapftests import yapf_test_helper
UNBREAKABLE = split_penalty.UNBREAKABLE
VERY_STRONGLY_CONNECTED = split_penalty.VERY_STRONGLY_CONNECTED
DOTTED_NAME = split_penalty.DOTTED_NAME
STRONGLY_CONNECTED = split_penalty.STRONGLY_CONNECTED

class SplitPenaltyTest(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        style.SetGlobalStyle(style.CreateYapfStyle())

    def _ParseAndComputePenalties(self, code, dumptree=False):
        if False:
            print('Hello World!')
        'Parses the code and computes split penalties.\n\n    Arguments:\n      code: code to parse as a string\n      dumptree: if True, the parsed pytree (after penalty assignment) is dumped\n        to stderr. Useful for debugging.\n\n    Returns:\n      Parse tree.\n    '
        tree = pytree_utils.ParseCodeToTree(code)
        split_penalty.ComputeSplitPenalties(tree)
        if dumptree:
            pytree_visitor.DumpPyTree(tree, target_stream=sys.stderr)
        return tree

    def _CheckPenalties(self, tree, list_of_expected):
        if False:
            return 10
        'Check that the tokens in the tree have the correct penalties.\n\n    Args:\n      tree: the pytree.\n      list_of_expected: list of (name, penalty) pairs. Non-semantic tokens are\n        filtered out from the expected values.\n    '

        def FlattenRec(tree):
            if False:
                i = 10
                return i + 15
            if pytree_utils.NodeName(tree) in pytree_utils.NONSEMANTIC_TOKENS:
                return []
            if isinstance(tree, pytree.Leaf):
                return [(tree.value, pytree_utils.GetNodeAnnotation(tree, pytree_utils.Annotation.SPLIT_PENALTY))]
            nodes = []
            for node in tree.children:
                nodes += FlattenRec(node)
            return nodes
        self.assertEqual(list_of_expected, FlattenRec(tree))

    def testUnbreakable(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        def foo(x):\n          pass\n    ')
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('def', None), ('foo', UNBREAKABLE), ('(', UNBREAKABLE), ('x', None), (')', STRONGLY_CONNECTED), (':', UNBREAKABLE), ('pass', None)])
        code = textwrap.dedent('        def foo(x):  # trailing comment\n          pass\n    ')
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('def', None), ('foo', UNBREAKABLE), ('(', UNBREAKABLE), ('x', None), (')', STRONGLY_CONNECTED), (':', UNBREAKABLE), ('pass', None)])
        code = textwrap.dedent('        class A:\n          pass\n        class B(A):\n          pass\n    ')
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('class', None), ('A', UNBREAKABLE), (':', UNBREAKABLE), ('pass', None), ('class', None), ('B', UNBREAKABLE), ('(', UNBREAKABLE), ('A', None), (')', None), (':', UNBREAKABLE), ('pass', None)])
        code = textwrap.dedent('        lambda a, b: None\n    ')
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('lambda', None), ('a', VERY_STRONGLY_CONNECTED), (',', VERY_STRONGLY_CONNECTED), ('b', VERY_STRONGLY_CONNECTED), (':', VERY_STRONGLY_CONNECTED), ('None', VERY_STRONGLY_CONNECTED)])
        code = textwrap.dedent('        import a.b.c\n    ')
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('import', None), ('a', None), ('.', UNBREAKABLE), ('b', UNBREAKABLE), ('.', UNBREAKABLE), ('c', UNBREAKABLE)])

    def testStronglyConnected(self):
        if False:
            return 10
        code = textwrap.dedent("        a = {\n            'x': 42,\n            y(lambda a: 23): 37,\n        }\n    ")
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('a', None), ('=', None), ('{', None), ("'x'", None), (':', STRONGLY_CONNECTED), ('42', None), (',', None), ('y', None), ('(', UNBREAKABLE), ('lambda', STRONGLY_CONNECTED), ('a', VERY_STRONGLY_CONNECTED), (':', VERY_STRONGLY_CONNECTED), ('23', VERY_STRONGLY_CONNECTED), (')', VERY_STRONGLY_CONNECTED), (':', STRONGLY_CONNECTED), ('37', None), (',', None), ('}', None)])
        code = textwrap.dedent('        [a for a in foo if a.x == 37]\n    ')
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('[', None), ('a', None), ('for', 0), ('a', STRONGLY_CONNECTED), ('in', STRONGLY_CONNECTED), ('foo', STRONGLY_CONNECTED), ('if', 0), ('a', STRONGLY_CONNECTED), ('.', VERY_STRONGLY_CONNECTED), ('x', DOTTED_NAME), ('==', STRONGLY_CONNECTED), ('37', STRONGLY_CONNECTED), (']', None)])

    def testFuncCalls(self):
        if False:
            return 10
        code = textwrap.dedent('        foo(1, 2, 3)\n    ')
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('foo', None), ('(', UNBREAKABLE), ('1', None), (',', UNBREAKABLE), ('2', None), (',', UNBREAKABLE), ('3', None), (')', VERY_STRONGLY_CONNECTED)])
        code = textwrap.dedent('        foo.bar.baz(1, 2, 3)\n    ')
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('foo', None), ('.', VERY_STRONGLY_CONNECTED), ('bar', DOTTED_NAME), ('.', VERY_STRONGLY_CONNECTED), ('baz', DOTTED_NAME), ('(', STRONGLY_CONNECTED), ('1', None), (',', UNBREAKABLE), ('2', None), (',', UNBREAKABLE), ('3', None), (')', VERY_STRONGLY_CONNECTED)])
        code = textwrap.dedent('        max(i for i in xrange(10))\n    ')
        tree = self._ParseAndComputePenalties(code)
        self._CheckPenalties(tree, [('max', None), ('(', UNBREAKABLE), ('i', 0), ('for', 0), ('i', STRONGLY_CONNECTED), ('in', STRONGLY_CONNECTED), ('xrange', STRONGLY_CONNECTED), ('(', UNBREAKABLE), ('10', STRONGLY_CONNECTED), (')', VERY_STRONGLY_CONNECTED), (')', VERY_STRONGLY_CONNECTED)])
if __name__ == '__main__':
    unittest.main()