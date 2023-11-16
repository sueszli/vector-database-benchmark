"""Tests for yapf.pytree_utils."""
import unittest
from yapf_third_party._ylib2to3 import pygram
from yapf_third_party._ylib2to3 import pytree
from yapf_third_party._ylib2to3.pgen2 import token
from yapf.pytree import pytree_utils
from yapftests import yapf_test_helper
_GRAMMAR_SYMBOL2NUMBER = pygram.python_grammar.symbol2number
_FOO = 'foo'
_FOO1 = 'foo1'
_FOO2 = 'foo2'
_FOO3 = 'foo3'
_FOO4 = 'foo4'
_FOO5 = 'foo5'

class NodeNameTest(yapf_test_helper.YAPFTest):

    def testNodeNameForLeaf(self):
        if False:
            return 10
        leaf = pytree.Leaf(token.LPAR, '(')
        self.assertEqual('LPAR', pytree_utils.NodeName(leaf))

    def testNodeNameForNode(self):
        if False:
            while True:
                i = 10
        leaf = pytree.Leaf(token.LPAR, '(')
        node = pytree.Node(pygram.python_grammar.symbol2number['suite'], [leaf])
        self.assertEqual('suite', pytree_utils.NodeName(node))

class ParseCodeToTreeTest(yapf_test_helper.YAPFTest):

    def testParseCodeToTree(self):
        if False:
            print('Hello World!')
        tree = pytree_utils.ParseCodeToTree('foo = 2\n')
        self.assertEqual('file_input', pytree_utils.NodeName(tree))
        self.assertEqual(2, len(tree.children))
        self.assertEqual('simple_stmt', pytree_utils.NodeName(tree.children[0]))

    def testPrintFunctionToTree(self):
        if False:
            i = 10
            return i + 15
        tree = pytree_utils.ParseCodeToTree('print("hello world", file=sys.stderr)\n')
        self.assertEqual('file_input', pytree_utils.NodeName(tree))
        self.assertEqual(2, len(tree.children))
        self.assertEqual('simple_stmt', pytree_utils.NodeName(tree.children[0]))

    def testPrintStatementToTree(self):
        if False:
            return 10
        with self.assertRaises(SyntaxError):
            pytree_utils.ParseCodeToTree('print "hello world"\n')

    def testClassNotLocal(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(SyntaxError):
            pytree_utils.ParseCodeToTree('class nonlocal: pass\n')

class InsertNodesBeforeAfterTest(yapf_test_helper.YAPFTest):

    def _BuildSimpleTree(self):
        if False:
            i = 10
            return i + 15
        lpar1 = pytree.Leaf(token.LPAR, '(')
        lpar2 = pytree.Leaf(token.LPAR, '(')
        simple_stmt = pytree.Node(_GRAMMAR_SYMBOL2NUMBER['simple_stmt'], [pytree.Leaf(token.NAME, 'foo')])
        return pytree.Node(_GRAMMAR_SYMBOL2NUMBER['suite'], [lpar1, lpar2, simple_stmt])

    def _MakeNewNodeRPAR(self):
        if False:
            while True:
                i = 10
        return pytree.Leaf(token.RPAR, ')')

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._simple_tree = self._BuildSimpleTree()

    def testInsertNodesBefore(self):
        if False:
            i = 10
            return i + 15
        pytree_utils.InsertNodesBefore([self._MakeNewNodeRPAR()], self._simple_tree.children[2])
        self.assertEqual(4, len(self._simple_tree.children))
        self.assertEqual('RPAR', pytree_utils.NodeName(self._simple_tree.children[2]))
        self.assertEqual('simple_stmt', pytree_utils.NodeName(self._simple_tree.children[3]))

    def testInsertNodesBeforeFirstChild(self):
        if False:
            for i in range(10):
                print('nop')
        simple_stmt = self._simple_tree.children[2]
        foo_child = simple_stmt.children[0]
        pytree_utils.InsertNodesBefore([self._MakeNewNodeRPAR()], foo_child)
        self.assertEqual(3, len(self._simple_tree.children))
        self.assertEqual(2, len(simple_stmt.children))
        self.assertEqual('RPAR', pytree_utils.NodeName(simple_stmt.children[0]))
        self.assertEqual('NAME', pytree_utils.NodeName(simple_stmt.children[1]))

    def testInsertNodesAfter(self):
        if False:
            i = 10
            return i + 15
        pytree_utils.InsertNodesAfter([self._MakeNewNodeRPAR()], self._simple_tree.children[2])
        self.assertEqual(4, len(self._simple_tree.children))
        self.assertEqual('simple_stmt', pytree_utils.NodeName(self._simple_tree.children[2]))
        self.assertEqual('RPAR', pytree_utils.NodeName(self._simple_tree.children[3]))

    def testInsertNodesAfterLastChild(self):
        if False:
            return 10
        simple_stmt = self._simple_tree.children[2]
        foo_child = simple_stmt.children[0]
        pytree_utils.InsertNodesAfter([self._MakeNewNodeRPAR()], foo_child)
        self.assertEqual(3, len(self._simple_tree.children))
        self.assertEqual(2, len(simple_stmt.children))
        self.assertEqual('NAME', pytree_utils.NodeName(simple_stmt.children[0]))
        self.assertEqual('RPAR', pytree_utils.NodeName(simple_stmt.children[1]))

    def testInsertNodesWhichHasParent(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(RuntimeError):
            pytree_utils.InsertNodesAfter([self._simple_tree.children[1]], self._simple_tree.children[0])

class AnnotationsTest(yapf_test_helper.YAPFTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._leaf = pytree.Leaf(token.LPAR, '(')
        self._node = pytree.Node(_GRAMMAR_SYMBOL2NUMBER['simple_stmt'], [pytree.Leaf(token.NAME, 'foo')])

    def testGetWhenNone(self):
        if False:
            print('Hello World!')
        self.assertIsNone(pytree_utils.GetNodeAnnotation(self._leaf, _FOO))

    def testSetWhenNone(self):
        if False:
            print('Hello World!')
        pytree_utils.SetNodeAnnotation(self._leaf, _FOO, 20)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._leaf, _FOO), 20)

    def testSetAgain(self):
        if False:
            return 10
        pytree_utils.SetNodeAnnotation(self._leaf, _FOO, 20)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._leaf, _FOO), 20)
        pytree_utils.SetNodeAnnotation(self._leaf, _FOO, 30)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._leaf, _FOO), 30)

    def testMultiple(self):
        if False:
            print('Hello World!')
        pytree_utils.SetNodeAnnotation(self._leaf, _FOO, 20)
        pytree_utils.SetNodeAnnotation(self._leaf, _FOO1, 1)
        pytree_utils.SetNodeAnnotation(self._leaf, _FOO2, 2)
        pytree_utils.SetNodeAnnotation(self._leaf, _FOO3, 3)
        pytree_utils.SetNodeAnnotation(self._leaf, _FOO4, 4)
        pytree_utils.SetNodeAnnotation(self._leaf, _FOO5, 5)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._leaf, _FOO), 20)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._leaf, _FOO1), 1)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._leaf, _FOO2), 2)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._leaf, _FOO3), 3)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._leaf, _FOO4), 4)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._leaf, _FOO5), 5)

    def testSubtype(self):
        if False:
            return 10
        pytree_utils.AppendNodeAnnotation(self._leaf, pytree_utils.Annotation.SUBTYPE, _FOO)
        self.assertSetEqual(pytree_utils.GetNodeAnnotation(self._leaf, pytree_utils.Annotation.SUBTYPE), {_FOO})
        pytree_utils.RemoveSubtypeAnnotation(self._leaf, _FOO)
        self.assertSetEqual(pytree_utils.GetNodeAnnotation(self._leaf, pytree_utils.Annotation.SUBTYPE), set())

    def testSetOnNode(self):
        if False:
            print('Hello World!')
        pytree_utils.SetNodeAnnotation(self._node, _FOO, 20)
        self.assertEqual(pytree_utils.GetNodeAnnotation(self._node, _FOO), 20)
if __name__ == '__main__':
    unittest.main()