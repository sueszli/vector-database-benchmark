"""Tests for utils.py."""
import textwrap
from pytype import debug
from pytype.typegraph import cfg
import unittest

class Node:
    """A graph node, for testing tree printing."""

    def __init__(self, name, *incoming):
        if False:
            return 10
        self.name = name
        self.outgoing = []
        self.incoming = list(incoming)
        for n in incoming:
            n.outgoing.append(self)

    def connect_to(self, other_node):
        if False:
            return 10
        self.outgoing.append(other_node)
        other_node.incoming.append(self)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'Node({self.name})'

class DebugTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.prog = cfg.Program()
        self.current_location = self.prog.NewCFGNode()

    def test_ascii_tree(self):
        if False:
            return 10
        n1 = Node('n1')
        n2 = Node('n2', n1)
        n3 = Node('n3', n2)
        n4 = Node('n4', n3)
        n5 = Node('n5', n1)
        n6 = Node('n6', n5)
        n7 = Node('n7', n5)
        del n4, n6
        s = debug.ascii_tree(n1, lambda n: n.outgoing)
        self.assertMultiLineEqual(textwrap.dedent('\n      Node(n1)\n      |\n      +-Node(n2)\n      | |\n      | +-Node(n3)\n      |   |\n      |   +-Node(n4)\n      |\n      +-Node(n5)\n        |\n        +-Node(n6)\n        |\n        +-Node(n7)\n    ').lstrip(), s)
        s = debug.ascii_tree(n7, lambda n: n.incoming)
        self.assertMultiLineEqual(textwrap.dedent('\n      Node(n7)\n      |\n      +-Node(n5)\n        |\n        +-Node(n1)\n    ').lstrip(), s)

    def test_ascii_graph(self):
        if False:
            for i in range(10):
                print('nop')
        n1 = Node('n1')
        n2 = Node('n2', n1)
        n3 = Node('n3', n2)
        n3.connect_to(n1)
        s = debug.ascii_tree(n1, lambda n: n.outgoing)
        self.assertMultiLineEqual(textwrap.dedent('\n      Node(n1)\n      |\n      +-Node(n2)\n        |\n        +-Node(n3)\n          |\n          +-[Node(n1)]\n    ').lstrip(), s)

    def test_ascii_graph_with_custom_text(self):
        if False:
            while True:
                i = 10
        n1 = Node('n1')
        n2 = Node('n2', n1)
        n3 = Node('n3', n2)
        n3.connect_to(n1)
        s = debug.ascii_tree(n1, lambda n: n.outgoing, lambda n: n.name.upper())
        self.assertMultiLineEqual(textwrap.dedent('\n      N1\n      |\n      +-N2\n        |\n        +-N3\n          |\n          +-[N1]\n    ').lstrip(), s)

    def test_root_cause(self):
        if False:
            while True:
                i = 10
        n1 = self.prog.NewCFGNode()
        n2 = self.prog.NewCFGNode()
        self.assertEqual((None, None), debug.root_cause([], n1))
        v = self.prog.NewVariable()
        b1 = v.AddBinding('foo', (), n2)
        self.assertEqual((b1, n1), debug.root_cause([b1], n1))
        v = self.prog.NewVariable()
        b2 = v.AddBinding('foo', (b1,), n1)
        self.assertEqual((b1, n1), debug.root_cause([b2], n1))

    def test_tree_pretty_printer(self):
        if False:
            print('Hello World!')
        n1 = self.prog.NewCFGNode('root')
        n2 = self.prog.NewCFGNode('init')
        n2.ConnectTo(n1)
        v = self.prog.NewVariable()
        b1 = v.AddBinding('foo', (), n2)
        v = self.prog.NewVariable()
        _ = v.AddBinding('bar', (b1,), n1)
        s = debug.prettyprint_cfg_tree(n1)
        assert isinstance(s, str)

    def test_pretty_print_binding_set(self):
        if False:
            print('Hello World!')
        v = self.prog.NewVariable()
        b1 = v.AddBinding('x', [], self.current_location)
        b2 = v.AddBinding('y', [], self.current_location)
        assert isinstance(debug.prettyprint_binding_set({b1, b2}), str)

    def test_pretty_print_binding_nested(self):
        if False:
            for i in range(10):
                print('nop')
        v1 = self.prog.NewVariable()
        b1 = v1.AddBinding('x', [], self.current_location)
        v2 = self.prog.NewVariable()
        b2 = v2.AddBinding('y', {b1}, self.current_location)
        assert isinstance(debug.prettyprint_binding_nested(b2), str)

    def test_program_to_text(self):
        if False:
            return 10
        v1 = self.prog.NewVariable()
        b = v1.AddBinding('x', [], self.current_location)
        n = self.current_location.ConnectNew()
        v2 = self.prog.NewVariable()
        v2.AddBinding('y', {b}, n)
        assert isinstance(debug.program_to_text(self.prog), str)

    def test_root_cause_visible(self):
        if False:
            return 10
        v = self.prog.NewVariable()
        b = v.AddBinding('x', [], self.current_location)
        self.assertEqual(debug.root_cause(b, self.current_location), (None, None))

    def test_root_cause_not_visible(self):
        if False:
            for i in range(10):
                print('nop')
        v = self.prog.NewVariable()
        b = v.AddBinding('x', [], self.current_location)
        n = self.current_location.ConnectNew()
        v.AddBinding('y', [], n)
        self.assertEqual(debug.root_cause(b, n), (b, n))
if __name__ == '__main__':
    unittest.main()