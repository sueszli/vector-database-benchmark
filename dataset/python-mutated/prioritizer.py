"""
Custom SPPF Prioritizer
=======================

This example demonstrates how to subclass ``ForestVisitor`` to make a custom
SPPF node prioritizer to be used in conjunction with ``TreeForestTransformer``.

Our prioritizer will count the number of descendants of a node that are tokens.
By negating this count, our prioritizer will prefer nodes with fewer token
descendants. Thus, we choose the more specific parse.
"""
from lark import Lark
from lark.parsers.earley_forest import ForestVisitor, TreeForestTransformer

class TokenPrioritizer(ForestVisitor):

    def visit_symbol_node_in(self, node):
        if False:
            for i in range(10):
                print('nop')
        return node.children

    def visit_packed_node_in(self, node):
        if False:
            i = 10
            return i + 15
        return node.children

    def visit_symbol_node_out(self, node):
        if False:
            for i in range(10):
                print('nop')
        priority = 0
        for child in node.children:
            priority += getattr(child, 'priority', -1)
        node.priority = priority

    def visit_packed_node_out(self, node):
        if False:
            print('Hello World!')
        priority = 0
        for child in node.children:
            priority += getattr(child, 'priority', -1)
        node.priority = priority

    def on_cycle(self, node, path):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Oops, we encountered a cycle.')
grammar = '\nstart: hello " " world | hello_world\nhello: "Hello"\nworld: "World"\nhello_world: "Hello World"\n'
parser = Lark(grammar, parser='earley', ambiguity='forest')
forest = parser.parse('Hello World')
print('Default prioritizer:')
tree = TreeForestTransformer(resolve_ambiguity=True).transform(forest)
print(tree.pretty())
forest = parser.parse('Hello World')
print('Custom prioritizer:')
tree = TreeForestTransformer(resolve_ambiguity=True, prioritizer=TokenPrioritizer()).transform(forest)
print(tree.pretty())