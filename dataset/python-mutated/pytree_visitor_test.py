"""Tests for yapf.pytree_visitor."""
import unittest
from io import StringIO
from yapf.pytree import pytree_utils
from yapf.pytree import pytree_visitor
from yapftests import yapf_test_helper

class _NodeNameCollector(pytree_visitor.PyTreeVisitor):
    """A tree visitor that collects the names of all tree nodes into a list.

  Attributes:
    all_node_names: collected list of the names, available when the traversal
      is over.
    name_node_values: collects a list of NAME leaves (in addition to those going
      into all_node_names).
  """

    def __init__(self):
        if False:
            print('Hello World!')
        self.all_node_names = []
        self.name_node_values = []

    def DefaultNodeVisit(self, node):
        if False:
            while True:
                i = 10
        self.all_node_names.append(pytree_utils.NodeName(node))
        super(_NodeNameCollector, self).DefaultNodeVisit(node)

    def DefaultLeafVisit(self, leaf):
        if False:
            print('Hello World!')
        self.all_node_names.append(pytree_utils.NodeName(leaf))

    def Visit_NAME(self, leaf):
        if False:
            for i in range(10):
                print('nop')
        self.name_node_values.append(leaf.value)
        self.DefaultLeafVisit(leaf)
_VISITOR_TEST_SIMPLE_CODE = 'foo = bar\nbaz = x\n'
_VISITOR_TEST_NESTED_CODE = 'if x:\n  if y:\n    return z\n'

class PytreeVisitorTest(yapf_test_helper.YAPFTest):

    def testCollectAllNodeNamesSimpleCode(self):
        if False:
            i = 10
            return i + 15
        tree = pytree_utils.ParseCodeToTree(_VISITOR_TEST_SIMPLE_CODE)
        collector = _NodeNameCollector()
        collector.Visit(tree)
        expected_names = ['file_input', 'simple_stmt', 'expr_stmt', 'NAME', 'EQUAL', 'NAME', 'NEWLINE', 'simple_stmt', 'expr_stmt', 'NAME', 'EQUAL', 'NAME', 'NEWLINE', 'ENDMARKER']
        self.assertEqual(expected_names, collector.all_node_names)
        expected_name_node_values = ['foo', 'bar', 'baz', 'x']
        self.assertEqual(expected_name_node_values, collector.name_node_values)

    def testCollectAllNodeNamesNestedCode(self):
        if False:
            while True:
                i = 10
        tree = pytree_utils.ParseCodeToTree(_VISITOR_TEST_NESTED_CODE)
        collector = _NodeNameCollector()
        collector.Visit(tree)
        expected_names = ['file_input', 'if_stmt', 'NAME', 'NAME', 'COLON', 'suite', 'NEWLINE', 'INDENT', 'if_stmt', 'NAME', 'NAME', 'COLON', 'suite', 'NEWLINE', 'INDENT', 'simple_stmt', 'return_stmt', 'NAME', 'NAME', 'NEWLINE', 'DEDENT', 'DEDENT', 'ENDMARKER']
        self.assertEqual(expected_names, collector.all_node_names)
        expected_name_node_values = ['if', 'x', 'if', 'y', 'return', 'z']
        self.assertEqual(expected_name_node_values, collector.name_node_values)

    def testDumper(self):
        if False:
            while True:
                i = 10
        tree = pytree_utils.ParseCodeToTree(_VISITOR_TEST_SIMPLE_CODE)
        stream = StringIO()
        pytree_visitor.PyTreeDumper(target_stream=stream).Visit(tree)
        dump_output = stream.getvalue()
        self.assertIn('file_input [3 children]', dump_output)
        self.assertIn("NAME(Leaf(NAME, 'foo'))", dump_output)
        self.assertIn("EQUAL(Leaf(EQUAL, '='))", dump_output)

    def testDumpPyTree(self):
        if False:
            i = 10
            return i + 15
        tree = pytree_utils.ParseCodeToTree(_VISITOR_TEST_SIMPLE_CODE)
        stream = StringIO()
        pytree_visitor.DumpPyTree(tree, target_stream=stream)
        dump_output = stream.getvalue()
        self.assertIn('file_input [3 children]', dump_output)
        self.assertIn("NAME(Leaf(NAME, 'foo'))", dump_output)
        self.assertIn("EQUAL(Leaf(EQUAL, '='))", dump_output)
if __name__ == '__main__':
    unittest.main()