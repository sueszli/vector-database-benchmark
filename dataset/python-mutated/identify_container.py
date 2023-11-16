"""Identify containers for lib2to3 trees.

This module identifies containers and the elements in them. Each element points
to the opening bracket and vice-versa.

  IdentifyContainers(): the main function exported by this module.
"""
from yapf_third_party._ylib2to3.pgen2 import token as grammar_token
from yapf.pytree import pytree_utils
from yapf.pytree import pytree_visitor

def IdentifyContainers(tree):
    if False:
        while True:
            i = 10
    'Run the identify containers visitor over the tree, modifying it in place.\n\n  Arguments:\n    tree: the top-level pytree node to annotate with subtypes.\n  '
    identify_containers = _IdentifyContainers()
    identify_containers.Visit(tree)

class _IdentifyContainers(pytree_visitor.PyTreeVisitor):
    """_IdentifyContainers - see file-level docstring for detailed description."""

    def Visit_trailer(self, node):
        if False:
            for i in range(10):
                print('nop')
        for child in node.children:
            self.Visit(child)
        if len(node.children) != 3:
            return
        if node.children[0].type != grammar_token.LPAR:
            return
        if pytree_utils.NodeName(node.children[1]) == 'arglist':
            for child in node.children[1].children:
                pytree_utils.SetOpeningBracket(pytree_utils.FirstLeafNode(child), node.children[0])
        else:
            pytree_utils.SetOpeningBracket(pytree_utils.FirstLeafNode(node.children[1]), node.children[0])

    def Visit_atom(self, node):
        if False:
            for i in range(10):
                print('nop')
        for child in node.children:
            self.Visit(child)
        if len(node.children) != 3:
            return
        if node.children[0].type != grammar_token.LPAR:
            return
        for child in node.children[1].children:
            pytree_utils.SetOpeningBracket(pytree_utils.FirstLeafNode(child), node.children[0])