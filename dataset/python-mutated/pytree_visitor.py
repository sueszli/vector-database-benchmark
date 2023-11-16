"""Generic visitor pattern for pytrees.

The lib2to3 parser produces a "pytree" - syntax tree consisting of Node
and Leaf types. This module implements a visitor pattern for such trees.

It also exports a basic "dumping" visitor that dumps a textual representation of
a pytree into a stream.

  PyTreeVisitor: a generic visitor pattern for pytrees.
  PyTreeDumper: a configurable "dumper" for displaying pytrees.
  DumpPyTree(): a convenience function to dump a pytree.
"""
import sys
from yapf_third_party._ylib2to3 import pytree
from yapf.pytree import pytree_utils

class PyTreeVisitor(object):
    """Visitor pattern for pytree trees.

  Methods named Visit_XXX will be invoked when a node with type XXX is
  encountered in the tree. The type is either a token type (for Leaf nodes) or
  grammar symbols (for Node nodes). The return value of Visit_XXX methods is
  ignored by the visitor.

  Visitors can modify node contents but must not change the tree structure
  (e.g. add/remove children and move nodes around).

  This is a very common visitor pattern in Python code; it's also used in the
  Python standard library ast module for providing AST visitors.

  Note: this makes names that aren't style conformant, so such visitor methods
  need to be marked with # pylint: disable=invalid-name We don't have a choice
  here, because lib2to3 nodes have under_separated names.

  For more complex behavior, the visit, DefaultNodeVisit and DefaultLeafVisit
  methods can be overridden. Don't forget to invoke DefaultNodeVisit for nodes
  that may have children - otherwise the children will not be visited.
  """

    def Visit(self, node):
        if False:
            while True:
                i = 10
        'Visit a node.'
        method = 'Visit_{0}'.format(pytree_utils.NodeName(node))
        if hasattr(self, method):
            getattr(self, method)(node)
        elif isinstance(node, pytree.Leaf):
            self.DefaultLeafVisit(node)
        else:
            self.DefaultNodeVisit(node)

    def DefaultNodeVisit(self, node):
        if False:
            i = 10
            return i + 15
        "Default visitor for Node: visits the node's children depth-first.\n\n    This method is invoked when no specific visitor for the node is defined.\n\n    Arguments:\n      node: the node to visit\n    "
        for child in node.children:
            self.Visit(child)

    def DefaultLeafVisit(self, leaf):
        if False:
            while True:
                i = 10
        'Default visitor for Leaf: no-op.\n\n    This method is invoked when no specific visitor for the leaf is defined.\n\n    Arguments:\n      leaf: the leaf to visit\n    '
        pass

def DumpPyTree(tree, target_stream=sys.stdout):
    if False:
        return 10
    'Convenience function for dumping a given pytree.\n\n  This function presents a very minimal interface. For more configurability (for\n  example, controlling how specific node types are displayed), use PyTreeDumper\n  directly.\n\n  Arguments:\n    tree: the tree to dump.\n    target_stream: the stream to dump the tree to. A file-like object. By\n      default will dump into stdout.\n  '
    dumper = PyTreeDumper(target_stream)
    dumper.Visit(tree)

class PyTreeDumper(PyTreeVisitor):
    """Visitor that dumps the tree to a stream.

  Implements the PyTreeVisitor interface.
  """

    def __init__(self, target_stream=sys.stdout):
        if False:
            while True:
                i = 10
        'Create a tree dumper.\n\n    Arguments:\n      target_stream: the stream to dump the tree to. A file-like object. By\n        default will dump into stdout.\n    '
        self._target_stream = target_stream
        self._current_indent = 0

    def _DumpString(self, s):
        if False:
            print('Hello World!')
        self._target_stream.write('{0}{1}\n'.format(' ' * self._current_indent, s))

    def DefaultNodeVisit(self, node):
        if False:
            return 10
        self._DumpString(pytree_utils.DumpNodeToString(node))
        self._current_indent += 2
        super(PyTreeDumper, self).DefaultNodeVisit(node)
        self._current_indent -= 2

    def DefaultLeafVisit(self, leaf):
        if False:
            i = 10
            return i + 15
        self._DumpString(pytree_utils.DumpNodeToString(leaf))