"""
Created on Aug 3, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import _ast
from ...asttools.visitors import Visitor

class Replacer(Visitor):
    """
    Visitor to replace nodes.
    """

    def __init__(self, old, new):
        if False:
            while True:
                i = 10
        self.old = old
        self.new = new

    def visitDefault(self, node):
        if False:
            i = 10
            return i + 15
        for field in node._fields:
            value = getattr(node, field)
            if value == self.old:
                setattr(node, field, self.new)
            if isinstance(value, (list, tuple)):
                for (i, item) in enumerate(value):
                    if item == self.old:
                        value[i] = self.new
                    elif isinstance(item, _ast.AST):
                        self.visit(item)
                    else:
                        pass
            elif isinstance(value, _ast.AST):
                self.visit(value)
            else:
                pass
        return

def replace_nodes(root, old, new):
    if False:
        print('Hello World!')
    '\n    Replace the old node with the new one.\n    Old must be an indirect child of root\n\n    :param root: ast node that contains an indirect reference to old\n    :param old: node to replace\n    :param new: node to replace `old` with\n    '
    rep = Replacer(old, new)
    rep.visit(root)
    return

class NodeRemover(Visitor):
    """
    Remove a node.
    """

    def __init__(self, to_remove):
        if False:
            while True:
                i = 10
        self.to_remove

    def visitDefault(self, node):
        if False:
            print('Hello World!')
        for field in node._fields:
            value = getattr(node, field)
            if value in self.to_remove:
                setattr(node, field, self.new)
            if isinstance(value, (list, tuple)):
                for (i, item) in enumerate(value):
                    if item == self.old:
                        value[i] = self.new
                    elif isinstance(item, _ast.AST):
                        self.visit(item)
                    else:
                        pass
            elif isinstance(value, _ast.AST):
                self.visit(value)
            else:
                pass
        return